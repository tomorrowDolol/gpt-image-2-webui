import OpenAI from "openai"
import { fetch as undiciFetch, ProxyAgent } from "undici"
import { NextResponse } from "next/server"

import { resolveLocale, t } from "@/lib/i18n"
import {
  extractGeneratedImages,
  getImageApiError,
  getPayloadField,
  normalizeImageEndpoint,
  normalizeOpenAIBaseURL,
} from "@/lib/image-request"

export const runtime = "nodejs"
export const maxDuration = 120

const MAX_IMAGE_BYTES = 10 * 1024 * 1024
const MIN_CUSTOM_DIMENSION = 64
const MAX_CUSTOM_DIMENSION = 8192
const GPT_IMAGE_SIZE_VALUES = new Set([
  "auto",
  "1024x1024",
  "1536x1024",
  "1024x1536",
  "2048x2048",
  "2048x1152",
  "3840x2160",
  "2160x3840",
])
const SUPPORTED_IMAGE_TYPES = new Set([
  "image/jpeg",
  "image/jpg",
  "image/png",
  "image/webp",
])
const OPENAI_PROXY_URL = process.env.OPENAI_PROXY_URL || process.env.HTTPS_PROXY || process.env.HTTP_PROXY || ""
const openAIProxyDispatcher = OPENAI_PROXY_URL ? new ProxyAgent(OPENAI_PROXY_URL) : undefined
const openAIProxyFetch: typeof fetch | undefined = openAIProxyDispatcher
  ? ((url: unknown, init?: unknown) =>
      undiciFetch(url as Parameters<typeof undiciFetch>[0], {
        ...(init as Record<string, unknown> | undefined),
        dispatcher: openAIProxyDispatcher,
      })) as unknown as typeof fetch
  : undefined

function getText(formData: FormData, key: string, fallback = "") {
  const value = formData.get(key)

  return typeof value === "string" && value.trim() ? value.trim() : fallback
}

function getBackground(formData: FormData) {
  const value = getText(formData, "background", "auto")
  return value === "transparent" || value === "opaque" || value === "auto" ? value : "auto"
}

function getOutputFormat(formData: FormData) {
  const value = getText(formData, "outputFormat", "png")
  return value === "jpeg" || value === "webp" || value === "png" ? value : "png"
}

function getGenerateQuality(formData: FormData) {
  const value = getText(formData, "quality", "auto")
  return value === "auto" || value === "low" || value === "medium" || value === "high" || value === "standard" || value === "hd"
    ? value
    : "auto"
}

function normalizeCustomSize(value: string) {
  const normalized = value.trim().toLowerCase().replace(/\s+/g, "").replace(/×/g, "x")
  const match = /^([1-9]\d{1,4})x([1-9]\d{1,4})$/.exec(normalized)

  if (!match) {
    return ""
  }

  const width = Number(match[1])
  const height = Number(match[2])

  if (
    width < MIN_CUSTOM_DIMENSION ||
    width > MAX_CUSTOM_DIMENSION ||
    height < MIN_CUSTOM_DIMENSION ||
    height > MAX_CUSTOM_DIMENSION
  ) {
    return ""
  }

  return `${width}x${height}`
}

function mapToNearestSupportedSize(size: string) {
  if (GPT_IMAGE_SIZE_VALUES.has(size)) {
    return size
  }

  const normalizedCustomSize = normalizeCustomSize(size)

  if (!normalizedCustomSize) {
    return "1024x1024"
  }

  const [width, height] = normalizedCustomSize.split("x").map(Number)
  const ratio = width / height

  if (ratio >= 0.75 && ratio <= 1.33) {
    return "1024x1024"
  }

  return ratio > 1.33 ? "1536x1024" : "1024x1536"
}

function getSize(formData: FormData) {
  return mapToNearestSupportedSize(getText(formData, "size", "1024x1024"))
}

function getEditQuality(formData: FormData) {
  const value = getText(formData, "quality", "auto")
  return value === "auto" || value === "low" || value === "medium" || value === "high" || value === "standard"
    ? value
    : "auto"
}

function getGenerateSize(formData: FormData) {
  return getSize(formData)
}

function getEditSize(formData: FormData) {
  return getSize(formData)
}

async function performImageEdit(
  editUrl: string,
  apiKey: string,
  prompt: string,
  model: string,
  n: number,
  outputFormat: string,
  background: string,
  quality: string,
  size: string,
  images: File[]
): Promise<unknown> {
  const { FormData: UndiciFormData } = await import("undici")
  const form = new UndiciFormData()
  form.append("model", model)
  form.append("prompt", prompt)
  form.append("n", String(n))
  form.append("output_format", outputFormat)
  form.append("background", background)
  form.append("quality", quality)
  form.append("size", size)

  for (const image of images) {
    const buffer = Buffer.from(await image.arrayBuffer())
    form.append("image", new Blob([buffer], { type: image.type }), image.name || "image.png")
  }

  const headers: Record<string, string> = {
    Authorization: `Bearer ${apiKey}`,
  }
  const rawForm = form as unknown as { headers: Record<string, string> | Headers }
  if (rawForm.headers instanceof Headers) {
    Object.assign(headers, Object.fromEntries(rawForm.headers.entries()))
  } else if (typeof rawForm.headers === "object" && rawForm.headers !== null) {
    Object.assign(headers, rawForm.headers as Record<string, string>)
  }

  const response = await undiciFetch(editUrl, {
    method: "POST",
    headers,
    body: form as unknown as BodyInit,
    dispatcher: openAIProxyDispatcher,
  } as Parameters<typeof undiciFetch>[1])

  if (!response.ok) {
    let errorBody: Record<string, unknown> = {}
    try {
      errorBody = (await response.json()) as Record<string, unknown>
    } catch {
      // ignore parse failure
    }
    const status = response.status
    const message = getImageApiError(errorBody) || `Request failed with status ${status}`
    const err = new Error(message) as Error & { status: number; errorBody: Record<string, unknown> }
    err.status = status
    err.errorBody = errorBody
    throw err
  }

  return response.json() as Promise<unknown>
}

export async function POST(request: Request) {
  let locale = resolveLocale(request.headers.get("accept-language"))
  let endpoint = ""

  try {
    const incomingFormData = await request.formData()
    locale = resolveLocale(
      ((): string => {
        const value = incomingFormData.get("locale")
        return typeof value === "string" && value.trim()
          ? value.trim()
          : request.headers.get("accept-language") || ""
      })()
    )
    const apiKey = getText(incomingFormData, "apiKey", process.env.OPENAI_API_KEY || "")
    const prompt = getText(incomingFormData, "prompt")

    if (!apiKey) {
      return NextResponse.json({ error: t(locale, "proxyApiKeyRequired") }, { status: 400 })
    }

    if (!prompt) {
      return NextResponse.json({ error: t(locale, "proxyPromptRequired") }, { status: 400 })
    }

    const images = incomingFormData
      .getAll("images")
      .filter((value): value is File => value instanceof File && value.size > 0)

    for (const image of images) {
      if (!SUPPORTED_IMAGE_TYPES.has(image.type)) {
        return NextResponse.json(
          { error: t(locale, "proxyUnsupportedImageFormat", { name: image.name }) },
          { status: 400 }
        )
      }

      if (image.size > MAX_IMAGE_BYTES) {
        return NextResponse.json(
          { error: t(locale, "proxyImageTooLarge", { name: image.name }) },
          { status: 400 }
        )
      }
    }

    const model = getText(incomingFormData, "model", "gpt-image-2")
    endpoint = normalizeImageEndpoint(getText(incomingFormData, "endpoint"), images.length > 0, locale)
    const baseURL = normalizeOpenAIBaseURL(getText(incomingFormData, "endpoint"), locale)
    const outputFormat = getOutputFormat(incomingFormData)
    const imageCount = Number(getText(incomingFormData, "imageCount", "1"))
    const background = getBackground(incomingFormData)
    const n = Math.min(Math.max(imageCount, 1), 4)
    let payload: unknown
    let requestQuality = "auto"
    let requestSize = "1024x1024"

    if (images.length > 0) {
      const quality = getEditQuality(incomingFormData)
      const size = getEditSize(incomingFormData)

      requestQuality = quality
      requestSize = size

      payload = await performImageEdit(
        endpoint,
        apiKey,
        prompt,
        model,
        n,
        outputFormat,
        background,
        quality,
        size,
        images
      )
    } else {
      const client = new OpenAI({
        apiKey,
        baseURL,
        maxRetries: 0,
        fetch: openAIProxyFetch,
      })
      const quality = getGenerateQuality(incomingFormData)
      const size = getGenerateSize(incomingFormData)

      requestQuality = quality
      requestSize = size
      payload = await client.images.generate({
        background,
        model,
        n,
        output_format: outputFormat,
        prompt,
        quality,
        size: size as OpenAI.Images.ImageGenerateParams["size"],
      })
    }

    const generatedImages = extractGeneratedImages(payload, outputFormat)

    if (!generatedImages.length) {
      return NextResponse.json(
        {
          endpoint,
          error: t(locale, "proxyNoImageField"),
        },
        { status: 502 }
      )
    }

    return NextResponse.json({
      background: getPayloadField(payload, "background"),
      created: getPayloadField(payload, "created"),
      endpoint,
      images: generatedImages,
      model,
      outputFormat,
      quality: getPayloadField(payload, "quality") || requestQuality,
      size: getPayloadField(payload, "size") || requestSize,
      usage: getPayloadField(payload, "usage"),
    })
  } catch (error) {
    if (error instanceof OpenAI.APIError) {
      return NextResponse.json(
        {
          endpoint,
          error: getImageApiError(error.error) || error.message || t(locale, "proxyRequestFailed", { status: error.status || 500 }),
        },
        { status: error.status || 500 }
      )
    }

    const errStatus = (error as Error & { status?: number }).status
    if (typeof errStatus === "number" && errStatus >= 400 && errStatus < 600) {
      const errBody = (error as Error & { errorBody?: Record<string, unknown> }).errorBody
      return NextResponse.json(
        {
          endpoint,
          error: getImageApiError(errBody) || (error instanceof Error ? error.message : t(locale, "proxyRequestFailed", { status: errStatus })),
        },
        { status: errStatus }
      )
    }

    return NextResponse.json(
      {
        endpoint,
        error: error instanceof Error ? error.message : t(locale, "proxyGenerationFailed"),
      },
      { status: 500 }
    )
  }
}
