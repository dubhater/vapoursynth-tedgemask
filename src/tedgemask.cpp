#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <type_traits>

#include <VapourSynth.h>
#include <VSHelper.h>


enum MaskTypes {
    TwoPixel = 1,
    FourPixel = 2,
    SixPixel = 5
};


enum LinkModes {
    LinkNothing,
    LinkChromaToLuma,
    LinkEverything
};


template <typename PixelType, MaskTypes type, bool binarize>
static void detect_edges_scalar(const uint8_t *srcp8, uint8_t *dstp8, int stride, int width, int height, int64_t threshold64, float scale, int pixel_max) {
    const PixelType *srcp = (const PixelType *)srcp8;
    PixelType *dstp = (PixelType *)dstp8;
    stride /= sizeof(PixelType);

    typedef typename std::conditional<sizeof(PixelType) == 1, int32_t, int64_t>::type int32_or_64;

    int32_or_64 threshold = (int32_or_64)threshold64;

    // Number of pixels to skip at the edges of the image.
    const int skip = type == FourPixel ? 2 : 1;

    for (int y = 0; y < skip; y++) {
        memset(dstp, 0, width * sizeof(PixelType));

        srcp += stride;
        dstp += stride;
    }

    for (int y = skip; y < height - skip; y++) {
        memset(dstp, 0, skip * sizeof(PixelType));

        for (int x = skip; x < width - skip; x++) {
            int32_or_64 gx, gy;
            float divisor;

            int top = srcp[x - stride];
            int left = srcp[x - 1];
            int right = srcp[x + 1];
            int bottom = srcp[x + stride];

            if (type == TwoPixel) {
                gx = right - left;
                gy = top - bottom;
                divisor = 0.25f;
            } else if (type == FourPixel) {
                int top2 = srcp[x - 2 * stride];
                int left2 = srcp[x - 2];
                int right2 = srcp[x + 2];
                int bottom2 = srcp[x + 2 * stride];

                gx = 12 * (left2 - right2) + 74 * (right - left);
                gy = 12 * (bottom2 - top2) + 74 * (top - bottom);
                divisor = 0.0001f;
            } else if (type == SixPixel) {
                int top_left = srcp[x - stride - 1];
                int top_right = srcp[x - stride + 1];
                int bottom_left = srcp[x + stride - 1];
                int bottom_right = srcp[x + stride + 1];

                gx = top_right + 2 * right + bottom_right - top_left - 2 * left - bottom_left;
                gy = bottom_left + 2 * bottom + bottom_right - top_left - 2 * top - top_right;
                divisor = 1.0f;
            }

            int32_or_64 sum_squares = gx * gx + gy * gy;

            if (binarize) {
                dstp[x] = (sum_squares > threshold) ? pixel_max : 0;
            } else {
                dstp[x] = std::min((int)(std::sqrt(sum_squares * divisor) * scale + 0.5f), pixel_max);
            }
        }

        memset(dstp + width - skip, 0, skip * sizeof(PixelType));

        srcp += stride;
        dstp += stride;
    }

    for (int y = height - skip; y < height; y++) {
        memset(dstp, 0, width * sizeof(PixelType));

        srcp += stride;
        dstp += stride;
    }
}


#if defined(TEDGEMASK_X86)

#include <emmintrin.h>


#if defined(_WIN32)
#define FORCE_INLINE __forceinline
#else
#define FORCE_INLINE inline __attribute__((always_inline))
#endif


#define zeroes _mm_setzero_si128()


template <typename PixelType, MaskTypes type, bool binarize>
static FORCE_INLINE void detect_edges_uint8_mmword_sse2(const PixelType *srcp, PixelType *dstp, int x, int stride, const __m128i &threshold, float scale) {
    __m128i gx, gy;
    __m128 divisor;

    __m128i top = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)&srcp[x - stride]),
                                    zeroes);
    __m128i left = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)&srcp[x - 1]),
                                     zeroes);
    __m128i right = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)&srcp[x + 1]),
                                      zeroes);
    __m128i bottom = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)&srcp[x + stride]),
                                       zeroes);

    if (type == TwoPixel) {
        gx = _mm_sub_epi16(right, left);
        gy = _mm_sub_epi16(top, bottom);
        divisor = _mm_set1_ps(0.25f);
    } else if (type == FourPixel) {
        __m128i top2 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)&srcp[x - 2 * stride]),
                                         zeroes);
        __m128i left2 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)&srcp[x - 2]),
                                          zeroes);
        __m128i right2 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)&srcp[x + 2]),
                                           zeroes);
        __m128i bottom2 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)&srcp[x + 2 * stride]),
                                            zeroes);

        gx = _mm_add_epi16(_mm_mullo_epi16(_mm_set1_epi16(12),
                                           _mm_sub_epi16(left2, right2)),
                           _mm_mullo_epi16(_mm_set1_epi16(74),
                                           _mm_sub_epi16(right, left)));

        gy = _mm_add_epi16(_mm_mullo_epi16(_mm_set1_epi16(12),
                                           _mm_sub_epi16(bottom2, top2)),
                           _mm_mullo_epi16(_mm_set1_epi16(74),
                                           _mm_sub_epi16(top, bottom)));

        divisor = _mm_set1_ps(0.0001f);
    } else if (type == SixPixel) {
        __m128i top_left = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)&srcp[x - stride - 1]),
                                             zeroes);
        __m128i top_right = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)&srcp[x - stride + 1]),
                                              zeroes);
        __m128i bottom_left = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)&srcp[x + stride - 1]),
                                                zeroes);
        __m128i bottom_right = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)&srcp[x + stride + 1]),
                                                 zeroes);

        __m128i sub_right_left = _mm_sub_epi16(right, left);
        __m128i sub_bottom_top = _mm_sub_epi16(bottom, top);

        gx = _mm_add_epi16(_mm_add_epi16(sub_right_left, sub_right_left),
                           _mm_add_epi16(_mm_sub_epi16(top_right, top_left),
                                         _mm_sub_epi16(bottom_right, bottom_left)));

        gy = _mm_add_epi16(_mm_add_epi16(sub_bottom_top, sub_bottom_top),
                           _mm_add_epi16(_mm_sub_epi16(bottom_left, top_left),
                                         _mm_sub_epi16(bottom_right, top_right)));

        divisor = _mm_set1_ps(1.0f);
    }

    __m128i gx_gy_lo = _mm_unpacklo_epi16(gx, gy);
    __m128i gx_gy_hi = _mm_unpackhi_epi16(gx, gy);
    __m128i sum_squares_lo = _mm_madd_epi16(gx_gy_lo, gx_gy_lo);
    __m128i sum_squares_hi = _mm_madd_epi16(gx_gy_hi, gx_gy_hi);

    __m128i output;

    if (binarize) {
        output = _mm_packs_epi16(_mm_packs_epi32(_mm_cmpgt_epi32(sum_squares_lo, threshold),
                                                 _mm_cmpgt_epi32(sum_squares_hi, threshold)),
                                 zeroes);
    } else {
        __m128 output_lo = _mm_add_ps(_mm_mul_ps(_mm_sqrt_ps(_mm_mul_ps(_mm_cvtepi32_ps(sum_squares_lo),
                                                                        divisor)),
                                                 _mm_set1_ps(scale)),
                                      _mm_set1_ps(0.5f));

        __m128 output_hi = _mm_add_ps(_mm_mul_ps(_mm_sqrt_ps(_mm_mul_ps(_mm_cvtepi32_ps(sum_squares_hi),
                                                                        divisor)),
                                                 _mm_set1_ps(scale)),
                                      _mm_set1_ps(0.5f));

        output = _mm_packus_epi16(_mm_packs_epi32(_mm_cvttps_epi32(output_lo),
                                                  _mm_cvttps_epi32(output_hi)),
                                  zeroes);
    }

    _mm_storel_epi64((__m128i *)&dstp[x], output);
}


template <typename PixelType, MaskTypes type, bool binarize>
static FORCE_INLINE void detect_edges_uint16_mmword_sse2(const PixelType *srcp, PixelType *dstp, int x, int stride, const __m128 &threshold, float scale, int pixel_max) {
    __m128 gx, gy, divisor;

    __m128i top = _mm_unpacklo_epi16(_mm_loadl_epi64((const __m128i *)&srcp[x - stride]),
                                     zeroes);
    __m128i left = _mm_unpacklo_epi16(_mm_loadl_epi64((const __m128i *)&srcp[x - 1]),
                                      zeroes);
    __m128i right = _mm_unpacklo_epi16(_mm_loadl_epi64((const __m128i *)&srcp[x + 1]),
                                       zeroes);
    __m128i bottom = _mm_unpacklo_epi16(_mm_loadl_epi64((const __m128i *)&srcp[x + stride]),
                                        zeroes);

    if (type == TwoPixel) {
        gx = _mm_cvtepi32_ps(_mm_sub_epi32(right, left));
        gy = _mm_cvtepi32_ps(_mm_sub_epi32(top, bottom));
        divisor = _mm_set1_ps(0.25f);
    } else if (type == FourPixel) {
        __m128i top2 = _mm_unpacklo_epi16(_mm_loadl_epi64((const __m128i *)&srcp[x - 2 * stride]),
                                          zeroes);
        __m128i left2 = _mm_unpacklo_epi16(_mm_loadl_epi64((const __m128i *)&srcp[x - 2]),
                                           zeroes);
        __m128i right2 = _mm_unpacklo_epi16(_mm_loadl_epi64((const __m128i *)&srcp[x + 2]),
                                            zeroes);
        __m128i bottom2 = _mm_unpacklo_epi16(_mm_loadl_epi64((const __m128i *)&srcp[x + 2 * stride]),
                                             zeroes);

        __m128i sub_left2_right2 = _mm_sub_epi32(left2, right2);
        __m128i sub_right_left = _mm_sub_epi32(right, left);

        gx = _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(sub_left2_right2),
                                   _mm_set1_ps(12.0f)),
                        _mm_mul_ps(_mm_cvtepi32_ps(sub_right_left),
                                   _mm_set1_ps(74.0f)));

        __m128i sub_bottom2_top2 = _mm_sub_epi32(bottom2, top2);
        __m128i sub_top_bottom = _mm_sub_epi32(top, bottom);

        gy = _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(sub_bottom2_top2),
                                   _mm_set1_ps(12.0f)),
                        _mm_mul_ps(_mm_cvtepi32_ps(sub_top_bottom),
                                   _mm_set1_ps(74.0f)));

        divisor = _mm_set1_ps(0.0001f);
    } else if (type == SixPixel) {
        __m128i top_left = _mm_unpacklo_epi16(_mm_loadl_epi64((const __m128i *)&srcp[x - stride - 1]),
                                              zeroes);
        __m128i top_right = _mm_unpacklo_epi16(_mm_loadl_epi64((const __m128i *)&srcp[x - stride + 1]),
                                               zeroes);
        __m128i bottom_left = _mm_unpacklo_epi16(_mm_loadl_epi64((const __m128i *)&srcp[x + stride - 1]),
                                                 zeroes);
        __m128i bottom_right = _mm_unpacklo_epi16(_mm_loadl_epi64((const __m128i *)&srcp[x + stride + 1]),
                                                  zeroes);

        __m128i sub_right_left = _mm_sub_epi32(right, left);
        __m128i sub_bottom_top = _mm_sub_epi32(bottom, top);

        gx = _mm_cvtepi32_ps(_mm_add_epi32(_mm_add_epi32(sub_right_left, sub_right_left),
                                           _mm_add_epi32(_mm_sub_epi32(top_right, top_left),
                                                         _mm_sub_epi32(bottom_right, bottom_left))));

        gy = _mm_cvtepi32_ps(_mm_add_epi32(_mm_add_epi32(sub_bottom_top, sub_bottom_top),
                                           _mm_add_epi32(_mm_sub_epi32(bottom_left, top_left),
                                                         _mm_sub_epi32(bottom_right, top_right))));

        divisor = _mm_set1_ps(1.0f);
    }

    __m128 sum_squares = _mm_add_ps(_mm_mul_ps(gx, gx),
                                    _mm_mul_ps(gy, gy));

    __m128i output;

    if (binarize) {
        output = _mm_packs_epi32(_mm_castps_si128(_mm_cmpnle_ps(sum_squares, threshold)),
                                 zeroes);
        output = _mm_and_si128(output,
                               _mm_set1_epi16(pixel_max));
    } else {
        output = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(_mm_sqrt_ps(_mm_mul_ps(sum_squares, divisor)),
                                                        _mm_set1_ps(scale)),
                                             _mm_set1_ps(-32767.5f))); // 0.5 for rounding and -32768 for packing

        output = _mm_add_epi16(_mm_min_epi16(_mm_packs_epi32(output, output),
                                             _mm_set1_epi16(pixel_max - 32768)),
                               _mm_set1_epi16(32768));
    }

    _mm_storel_epi64((__m128i *)&dstp[x], output);
}


template <typename PixelType, MaskTypes type, bool binarize>
static void detect_edges_sse2(const uint8_t *srcp8, uint8_t *dstp8, int stride, int width, int height, int64_t threshold64, float scale, int pixel_max) {
    const PixelType *srcp = (const PixelType *)srcp8;
    PixelType *dstp = (PixelType *)dstp8;
    stride /= sizeof(PixelType);

    __m128i threshold_epi32 = _mm_set1_epi32((int)threshold64);
    __m128 threshold_ps = _mm_set1_ps((float)threshold64);

    // Number of pixels to skip at the edges of the image.
    const int skip = type == FourPixel ? 2 : 1;

    const int pixels_per_iteration = 8 / sizeof(PixelType);

    const int width_simd = (width - 2 * skip) / pixels_per_iteration * pixels_per_iteration + 2 * skip;

    for (int y = 0; y < skip; y++) {
        memset(dstp, 0, width * sizeof(PixelType));

        srcp += stride;
        dstp += stride;
    }

    for (int y = skip; y < height - skip; y++) {
        memset(dstp, 0, skip * sizeof(PixelType));

        for (int x = skip; x < width_simd - skip; x += pixels_per_iteration) {
            if (sizeof(PixelType) == 1)
                detect_edges_uint8_mmword_sse2<PixelType, type, binarize>(srcp, dstp, x, stride, threshold_epi32, scale);
            else
                detect_edges_uint16_mmword_sse2<PixelType, type, binarize>(srcp, dstp, x, stride, threshold_ps, scale, pixel_max);
        }

        if (width > width_simd) {
            if (sizeof(PixelType) == 1)
                detect_edges_uint8_mmword_sse2<PixelType, type, binarize>(srcp, dstp, width - skip - pixels_per_iteration, stride, threshold_epi32, scale);
            else
                detect_edges_uint16_mmword_sse2<PixelType, type, binarize>(srcp, dstp, width - skip - pixels_per_iteration, stride, threshold_ps, scale, pixel_max);
        }

        memset(dstp + width - skip, 0, skip * sizeof(PixelType));

        srcp += stride;
        dstp += stride;
    }

    for (int y = height - skip; y < height; y++) {
        memset(dstp, 0, width * sizeof(PixelType));

        srcp += stride;
        dstp += stride;
    }
}

#endif // defined(TEDGEMASK_X86)


template <typename PixelType, LinkModes link>
static void link_planes_444_scalar(uint8_t *dstp18, uint8_t *dstp28, uint8_t *dstp38, int stride1, int stride2, int width, int height, int pixel_max) {
    (void)stride2;
    (void)pixel_max;

    PixelType *dstp1 = (PixelType *)dstp18;
    PixelType *dstp2 = (PixelType *)dstp28;
    PixelType *dstp3 = (PixelType *)dstp38;
    stride1 /= sizeof(PixelType);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            PixelType val = dstp1[x];

            if (link == LinkEverything) {
                val |= dstp2[x] | dstp3[x];

                if (val)
                    dstp1[x] = val;
            }

            if (val)
                dstp2[x] = dstp3[x] = val;
        }

        dstp1 += stride1;
        dstp2 += stride1;
        dstp3 += stride1;
    }
}


template <typename PixelType, LinkModes link>
static void link_planes_422_scalar(uint8_t *dstp18, uint8_t *dstp28, uint8_t *dstp38, int stride1, int stride2, int width, int height, int pixel_max) {
    (void)pixel_max;

    PixelType *dstp1 = (PixelType *)dstp18;
    PixelType *dstp2 = (PixelType *)dstp28;
    PixelType *dstp3 = (PixelType *)dstp38;
    stride1 /= sizeof(PixelType);
    stride2 /= sizeof(PixelType);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x += 2) {
            PixelType val = dstp1[x] & dstp1[x + 1];

            if (link == LinkEverything) {
                val |= dstp2[x >> 1] | dstp3[x >> 1];

                if (val)
                    dstp1[x] = dstp1[x + 1] = val;
            }

            if (val)
                dstp2[x >> 1] = dstp3[x >> 1] = val;
        }

        dstp1 += stride1;
        dstp2 += stride2;
        dstp3 += stride2;
    }
}


template <typename PixelType, LinkModes link>
static void link_planes_440_scalar(uint8_t *dstp18, uint8_t *dstp28, uint8_t *dstp38, int stride1, int stride2, int width, int height, int pixel_max) {
    (void)stride2;
    (void)pixel_max;

    PixelType *dstp1 = (PixelType *)dstp18;
    PixelType *dstp2 = (PixelType *)dstp28;
    PixelType *dstp3 = (PixelType *)dstp38;
    stride1 /= sizeof(PixelType);

    for (int y = 0; y < height; y += 2) {
        for (int x = 0; x < width; x++) {
            PixelType val = dstp1[x] & dstp1[x + stride1];

            if (link == LinkEverything) {
                val |= dstp2[x] | dstp3[x];

                if (val)
                    dstp1[x] = dstp1[x + stride1] = val;
            }

            if (val)
                dstp2[x] = dstp3[x] = val;
        }

        dstp1 += stride1 * 2;
        dstp2 += stride1;
        dstp3 += stride1;
    }
}


template <typename PixelType, LinkModes link>
static void link_planes_420_scalar(uint8_t *dstp18, uint8_t *dstp28, uint8_t *dstp38, int stride1, int stride2, int width, int height, int pixel_max) {
    PixelType *dstp1 = (PixelType *)dstp18;
    PixelType *dstp2 = (PixelType *)dstp28;
    PixelType *dstp3 = (PixelType *)dstp38;
    stride1 /= sizeof(PixelType);
    stride2 /= sizeof(PixelType);

    for (int y = 0; y < height; y += 2) {
        for (int x = 0; x < width; x += 2) {
            int sum = 0;

            if (dstp1[x])
                sum++;
            if (dstp1[x + 1])
                sum++;
            if (dstp1[x + stride1])
                sum++;
            if (dstp1[x + stride1 + 1])
                sum++;

            if (link == LinkEverything) {
                if (dstp2[x >> 1])
                    sum += 2;
                if (dstp3[x >> 1])
                    sum += 2;

                if (sum > 1)
                    dstp1[x] = dstp1[x + 1] = dstp1[x + stride1] = dstp1[x + stride1 + 1] = pixel_max;
            }

            if (sum > 1)
                dstp2[x >> 1] = dstp3[x >> 1] = pixel_max;
        }

        dstp1 += stride1 * 2;
        dstp2 += stride2;
        dstp3 += stride2;
    }
}


typedef struct TEdgeMaskData {
    VSNodeRef *clip;
    const VSVideoInfo *vi;

    int64_t threshold[3];
    float scale;

    int process[3];

    decltype(detect_edges_scalar<uint8_t, TwoPixel, true>) *detect_edges[3];
    decltype(link_planes_444_scalar<uint8_t, LinkEverything>) *link_planes;
} TEdgeMaskData;


static void VS_CC TEdgeMaskInit(VSMap *in, VSMap *out, void **instanceData, VSNode *node, VSCore *core, const VSAPI *vsapi) {
    (void)in;
    (void)out;
    (void)core;

    TEdgeMaskData *d = (TEdgeMaskData *) *instanceData;

    vsapi->setVideoInfo(d->vi, 1, node);
}


static const VSFrameRef *VS_CC TEdgeMaskGetFrame(int n, int activationReason, void **instanceData, void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {
    (void)frameData;

    const TEdgeMaskData *d = (const TEdgeMaskData *) *instanceData;

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->clip, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const VSFrameRef *src = vsapi->getFrameFilter(n, d->clip, frameCtx);

        const VSFrameRef *plane_src[3] = {
            d->process[0] ? nullptr : src,
            d->process[1] ? nullptr : src,
            d->process[2] ? nullptr : src
        };

        int planes[3] = { 0, 1, 2 };

        VSFrameRef *dst = vsapi->newVideoFrame2(d->vi->format,
                                                vsapi->getFrameWidth(src, 0),
                                                vsapi->getFrameHeight(src, 0),
                                                plane_src,
                                                planes,
                                                src,
                                                core);

        int pixel_max = (1 << d->vi->format->bitsPerSample) - 1;

        for (int plane = 0; plane < d->vi->format->numPlanes; plane++) {
            if (!d->process[plane])
                continue;

            const uint8_t *srcp = vsapi->getReadPtr(src, plane);
            uint8_t *dstp = vsapi->getWritePtr(dst, plane);
            int stride = vsapi->getStride(src, plane);
            int width = vsapi->getFrameWidth(src, plane);
            int height = vsapi->getFrameHeight(src, plane);

            d->detect_edges[plane](srcp, dstp, stride, width, height, d->threshold[plane], d->scale, pixel_max);
        }

        vsapi->freeFrame(src);

        if (d->link_planes) {
            d->link_planes(vsapi->getWritePtr(dst, 0),
                           vsapi->getWritePtr(dst, 1),
                           vsapi->getWritePtr(dst, 2),
                           vsapi->getStride(dst, 0),
                           vsapi->getStride(dst, 1),
                           vsapi->getFrameWidth(dst, 0),
                           vsapi->getFrameHeight(dst, 0),
                           pixel_max);
        }

        return dst;
    }

    return nullptr;
}


static void VS_CC TEdgeMaskFree(void *instanceData, VSCore *core, const VSAPI *vsapi) {
    (void)core;

    TEdgeMaskData *d = (TEdgeMaskData *)instanceData;

    vsapi->freeNode(d->clip);
    free(d);
}


static void selectFunctions(TEdgeMaskData *d, int type, int link, int opt) {
    int bits = d->vi->format->bitsPerSample;

    for (int plane = 0; plane < d->vi->format->numPlanes; plane++) {
        if (bits == 8) {
            if (d->threshold[plane] == 0) {
                if (type == TwoPixel)
                    d->detect_edges[plane] = detect_edges_scalar<uint8_t, TwoPixel, false>;
                else if (type == FourPixel)
                    d->detect_edges[plane] = detect_edges_scalar<uint8_t, FourPixel, false>;
                else if (type == SixPixel)
                    d->detect_edges[plane] = detect_edges_scalar<uint8_t, SixPixel, false>;
            } else {
                if (type == TwoPixel)
                    d->detect_edges[plane] = detect_edges_scalar<uint8_t, TwoPixel, true>;
                else if (type == FourPixel)
                    d->detect_edges[plane] = detect_edges_scalar<uint8_t, FourPixel, true>;
                else if (type == SixPixel)
                    d->detect_edges[plane] = detect_edges_scalar<uint8_t, SixPixel, true>;
            }
        } else {
            if (d->threshold[plane] == 0) {
                if (type == TwoPixel)
                    d->detect_edges[plane] = detect_edges_scalar<uint16_t, TwoPixel, false>;
                else if (type == FourPixel)
                    d->detect_edges[plane] = detect_edges_scalar<uint16_t, FourPixel, false>;
                else if (type == SixPixel)
                    d->detect_edges[plane] = detect_edges_scalar<uint16_t, SixPixel, false>;
            } else {
                if (type == TwoPixel)
                    d->detect_edges[plane] = detect_edges_scalar<uint16_t, TwoPixel, true>;
                else if (type == FourPixel)
                    d->detect_edges[plane] = detect_edges_scalar<uint16_t, FourPixel, true>;
                else if (type == SixPixel)
                    d->detect_edges[plane] = detect_edges_scalar<uint16_t, SixPixel, true>;
            }
        }

#if defined(TEDGEMASK_X86)
        if (opt) {
            if (bits == 8) {
                if (d->threshold[plane] == 0) {
                    if (type == TwoPixel)
                        d->detect_edges[plane] = detect_edges_sse2<uint8_t, TwoPixel, false>;
                    else if (type == FourPixel)
                        d->detect_edges[plane] = detect_edges_sse2<uint8_t, FourPixel, false>;
                    else if (type == SixPixel)
                        d->detect_edges[plane] = detect_edges_sse2<uint8_t, SixPixel, false>;
                } else {
                    if (type == TwoPixel)
                        d->detect_edges[plane] = detect_edges_sse2<uint8_t, TwoPixel, true>;
                    else if (type == FourPixel)
                        d->detect_edges[plane] = detect_edges_sse2<uint8_t, FourPixel, true>;
                    else if (type == SixPixel)
                        d->detect_edges[plane] = detect_edges_sse2<uint8_t, SixPixel, true>;
                }
            } else {
                if (d->threshold[plane] == 0) {
                    if (type == TwoPixel)
                        d->detect_edges[plane] = detect_edges_sse2<uint16_t, TwoPixel, false>;
                    else if (type == FourPixel)
                        d->detect_edges[plane] = detect_edges_sse2<uint16_t, FourPixel, false>;
                    else if (type == SixPixel)
                        d->detect_edges[plane] = detect_edges_sse2<uint16_t, SixPixel, false>;
                } else {
                    if (type == TwoPixel)
                        d->detect_edges[plane] = detect_edges_sse2<uint16_t, TwoPixel, true>;
                    else if (type == FourPixel)
                        d->detect_edges[plane] = detect_edges_sse2<uint16_t, FourPixel, true>;
                    else if (type == SixPixel)
                        d->detect_edges[plane] = detect_edges_sse2<uint16_t, SixPixel, true>;
                }
            }
        }
#endif
    }

    if (d->vi->format->subSamplingW == 0) {
        if (d->vi->format->subSamplingH == 0) {
            if (bits == 8) {
                if (link == LinkChromaToLuma)
                    d->link_planes = link_planes_444_scalar<uint8_t, LinkChromaToLuma>;
                else if (link == LinkEverything)
                    d->link_planes = link_planes_444_scalar<uint8_t, LinkEverything>;
            } else {
                if (link == LinkChromaToLuma)
                    d->link_planes = link_planes_444_scalar<uint16_t, LinkChromaToLuma>;
                else if (link == LinkEverything)
                    d->link_planes = link_planes_444_scalar<uint16_t, LinkEverything>;
            }
        } else {
            if (bits == 8) {
                if (link == LinkChromaToLuma)
                    d->link_planes = link_planes_440_scalar<uint8_t, LinkChromaToLuma>;
                else if (link == LinkEverything)
                    d->link_planes = link_planes_440_scalar<uint8_t, LinkEverything>;
            } else {
                if (link == LinkChromaToLuma)
                    d->link_planes = link_planes_440_scalar<uint16_t, LinkChromaToLuma>;
                else if (link == LinkEverything)
                    d->link_planes = link_planes_440_scalar<uint16_t, LinkEverything>;
            }
        }
    } else {
        if (d->vi->format->subSamplingH == 0) {
            if (bits == 8) {
                if (link == LinkChromaToLuma)
                    d->link_planes = link_planes_422_scalar<uint8_t, LinkChromaToLuma>;
                else if (link == LinkEverything)
                    d->link_planes = link_planes_422_scalar<uint8_t, LinkEverything>;
            } else {
                if (link == LinkChromaToLuma)
                    d->link_planes = link_planes_422_scalar<uint16_t, LinkChromaToLuma>;
                else if (link == LinkEverything)
                    d->link_planes = link_planes_422_scalar<uint16_t, LinkEverything>;
            }
        } else {
            if (bits == 8) {
                if (link == LinkChromaToLuma)
                    d->link_planes = link_planes_420_scalar<uint8_t, LinkChromaToLuma>;
                else if (link == LinkEverything)
                    d->link_planes = link_planes_420_scalar<uint8_t, LinkEverything>;
            } else {
                if (link == LinkChromaToLuma)
                    d->link_planes = link_planes_420_scalar<uint16_t, LinkChromaToLuma>;
                else if (link == LinkEverything)
                    d->link_planes = link_planes_420_scalar<uint16_t, LinkEverything>;
            }
        }
    }
}


static void VS_CC TEdgeMaskCreate(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) {
    (void)userData;

    TEdgeMaskData d;
    memset(&d, 0, sizeof(d));

    int err;

    int type = int64ToIntS(vsapi->propGetInt(in, "type", 0, &err));
    if (err)
        type = FourPixel;

    if (type < 1 || type > 5) {
        vsapi->setError(out, "TEdgeMask: type must be between 1 and 5 (inclusive).");
        return;
    }

    // Types 3 and 4 from TEMmod are not implemented.
    // They are more or less types 1 and 2 but with integer SSE2 code.
    if (type == 3)
        type = TwoPixel;
    if (type == 4)
        type = FourPixel;


    d.clip = vsapi->propGetNode(in, "clip", 0, nullptr);
    d.vi = vsapi->getVideoInfo(d.clip);


    if (!d.vi->format || d.vi->format->sampleType != stInteger || d.vi->format->bitsPerSample > 16 || d.vi->format->subSamplingW > 1 || d.vi->format->subSamplingH > 1) {
        vsapi->setError(out, "TEdgeMask: clip must have constant format, 8..16 bit integer samples, and subsampling ratios of at most 2.");
        vsapi->freeNode(d.clip);
        return;
    }


    double th[3];

    for (int i = 0; i < 3; i++) {
        th[i] = vsapi->propGetFloat(in, "threshold", i, &err);
        if (err)
            th[i] = (i == 0) ? 8.0
                             : th[i - 1];

        if (th[i] < 0) {
            vsapi->setError(out, "TEdgeMask: threshold must not be negative.");
            vsapi->freeNode(d.clip);
            return;
        }
    }

    for (int i = 0; i < 3; i++) {
        th[i] *= 1 << (d.vi->format->bitsPerSample - 8);

        th[i] *= th[i];
        if (type == TwoPixel)
            th[i] *= 4;
        else if (type == FourPixel)
            th[i] *= 10000;
        else if (type == SixPixel)
            th[i] *= 16;

        d.threshold[i] = (int64_t)(th[i] + 0.5);
    }


    int n = d.vi->format->numPlanes;
    int m = vsapi->propNumElements(in, "planes");

    for (int i = 0; i < 3; i++)
        d.process[i] = (m <= 0);

    for (int i = 0; i < m; i++) {
        int o = int64ToIntS(vsapi->propGetInt(in, "planes", i, nullptr));

        if (o < 0 || o >= n) {
            vsapi->freeNode(d.clip);
            vsapi->setError(out, "TEdgeMask: plane index out of range");
            return;
        }

        if (d.process[o]) {
            vsapi->freeNode(d.clip);
            vsapi->setError(out, "TEdgeMask: plane specified twice");
            return;
        }

        d.process[o] = 1;
    }


    int link = int64ToIntS(vsapi->propGetInt(in, "link", 0, &err));
    if (err) {
        if (d.vi->format->colorFamily == cmRGB)
            link = LinkEverything;
        else
            link = LinkChromaToLuma;
    }

    if (link < 0 || link > 2) {
        vsapi->setError(out, "TEdgeMask: link must be 0, 1, or 2.");
        vsapi->freeNode(d.clip);
        return;
    }

    if (link == LinkChromaToLuma && d.vi->format->colorFamily == cmRGB) {
        vsapi->setError(out, "TEdgeMask: link must be 0 or 2 when clip is RGB.");
        vsapi->freeNode(d.clip);
        return;
    }

    if (d.vi->format->colorFamily == cmGray ||
        d.threshold[0] == 0 || d.threshold[1] == 0 || d.threshold[2] == 0 ||
        !d.process[0] || !d.process[1] || !d.process[2])
        link = LinkNothing;


    d.scale = (float)vsapi->propGetFloat(in, "scale", 0, &err);
    if (err)
        d.scale = 1.0f;

    if (d.scale < 0.0f) {
        vsapi->setError(out, "TEdgeMask: scale must not be negative.");
        vsapi->freeNode(d.clip);
        return;
    }

    if (type == TwoPixel)
        d.scale *= 255.0f / 127.5f;
    else if (type == FourPixel)
        d.scale *= 255.0f / 158.1f;
    else if (type == SixPixel)
        d.scale *= 0.25f;


    bool opt = !!vsapi->propGetInt(in, "opt", 0, &err);
    if (err)
        opt = true;


    selectFunctions(&d, type, link, opt);


    TEdgeMaskData *data = (TEdgeMaskData *)malloc(sizeof(d));
    *data = d;

    vsapi->createFilter(in, out, "TEdgeMask", TEdgeMaskInit, TEdgeMaskGetFrame, TEdgeMaskFree, fmParallel, 0, data, core);
}


VS_EXTERNAL_API(void) VapourSynthPluginInit(VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin *plugin) {
    configFunc("com.nodame.tedgemask", "tedgemask", "Edge detection plugin", VAPOURSYNTH_API_VERSION, 1, plugin);
    registerFunc("TEdgeMask",
                 "clip:clip;"
                 "threshold:float[]:opt;"
                 "type:int:opt;"
                 "link:int:opt;"
                 "scale:float:opt;"
                 "planes:int[]:opt;"
                 "opt:int:opt;"
                 , TEdgeMaskCreate, nullptr, plugin);
}
