#include "img_converters.h"
#include <AViShaESPCam.h>
#include <WebServer.h>
#include <WiFi.h>

AViShaESPCam cam;
WebServer server(80);

const char *ssid = "Proteus";
const char *password = "proteus0123456789";

#define FIRE_DEBUG true

// ── Bounding-box for the detected fire region ──────────────
struct BBox {
    int x1 = 9999, y1 = 9999;
    int x2 = 0, y2 = 0;
    bool valid = false;
};

// ── Fire detector ───────────────────────────────────────────
class FireDetector {
public:
    static constexpr float FIRE_PIXEL_RATIO = 0.004f;
    static constexpr uint8_t MIN_VALUE = 180;
    static constexpr uint8_t MIN_SATURATION = 80;
    static constexpr uint8_t MIN_RED_DOMINANCE = 30;
    static constexpr float FLICKER_THRESH = 0.000005f;
    static constexpr uint8_t CONFIRM_FRAMES = 3;
    static constexpr uint8_t HIST_LEN = 8;

    // Frame geometry – must match the init() resolution (VGA)
    static constexpr uint16_t FRAME_W = 640;
    static constexpr uint16_t FRAME_H = 480;
    static constexpr size_t RGB_SIZE = (size_t)FRAME_W * FRAME_H * 3;

    FireDetector() : _histIdx(0), _consecCount(0) {
        memset(_history, 0, sizeof(_history));
    }

    // ── Main update – decode + analyse + flicker ────────────
    bool update(const uint8_t *jpegBuf, size_t jpegLen) {
        float fireRatio = _analyseFrame(jpegBuf, jpegLen); // also fills _lastBBox

        _history[_histIdx] = fireRatio;
        _histIdx = (_histIdx + 1) % HIST_LEN;

        bool colorMatch = (fireRatio >= FIRE_PIXEL_RATIO);
        float var = _variance();
        bool flickerOk = (var >= FLICKER_THRESH);

#if FIRE_DEBUG
        Serial.printf("[DBG] ratio=%.5f (need>=%.3f) colorOK=%d | var=%.8f flickerOK=%d | consec=%d\n",
                      fireRatio, FIRE_PIXEL_RATIO, colorMatch,
                      var, flickerOk, _consecCount);
#endif

        if (colorMatch && flickerOk)
            _consecCount = min(_consecCount + 1, (int)CONFIRM_FRAMES);
        else
            _consecCount = max(_consecCount - 1, 0);

        _lastRatio = fireRatio;
        return (_consecCount >= CONFIRM_FRAMES);
    }

    float lastFireRatio() const { return _lastRatio; }
    int confidence() const { return _consecCount; }
    BBox lastBBox() const { return _lastBBox; }

    // ── Draw bounding box on the cached RGB buffer, re-encode
    // Caller must free(*out) with free() after use.
    bool getAnnotatedJpeg(uint8_t **out, size_t *outLen) {
        if (_rgb == nullptr)
            return false;
        if (_lastBBox.valid)
            _drawRect(_rgb, _lastBBox, /*r*/ 255, /*g*/ 64, /*b*/ 0, /*thickness*/ 3);
        return fmt2jpg(_rgb, RGB_SIZE, FRAME_W, FRAME_H,
                       PIXFORMAT_RGB888, /*quality*/ 80, out, outLen);
    }

private:
    float _history[HIST_LEN];
    uint8_t _histIdx;
    int _consecCount;
    float _lastRatio = 0.0f;
    BBox _lastBBox;
    uint8_t *_rgb = nullptr; // lives in PSRAM, allocated once

    // ── Per-frame analysis ──────────────────────────────────
    float _analyseFrame(const uint8_t *jpegBuf, size_t jpegLen) {
        if (_rgb == nullptr) {
            _rgb = (uint8_t *)ps_malloc(RGB_SIZE);
            if (_rgb == nullptr) {
                Serial.println("[FireDetector] ps_malloc failed!");
                return 0.0f;
            }
        }

        if (!fmt2rgb888(jpegBuf, jpegLen, PIXFORMAT_JPEG, _rgb)) {
            Serial.println("[FireDetector] fmt2rgb888 failed!");
            return 0.0f;
        }

        // Reset bounding box for this frame
        _lastBBox = {9999, 9999, 0, 0, false};

        uint32_t total = 0, fire = 0;
#if FIRE_DEBUG
        uint32_t rejDark = 0, rejGrey = 0, rejSat = 0, rejHue = 0;
        uint8_t sR = 0, sG = 0, sB = 0;
        bool gotSample = false;
#endif

        constexpr size_t STEP = 3 * 4; // sample every 4th pixel
        for (size_t i = 0; i + 2 < RGB_SIZE; i += STEP) {
            uint8_t r = _rgb[i], g = _rgb[i + 1], b = _rgb[i + 2];
            total++;

#if FIRE_DEBUG
            uint8_t maxC0 = max(r, max(g, b));
            if (!gotSample && maxC0 >= MIN_VALUE && r >= b + MIN_RED_DOMINANCE) {
                sR = r;
                sG = g;
                sB = b;
                gotSample = true;
            }
            if (maxC0 < MIN_VALUE) {
                rejDark++;
                continue;
            }
            uint8_t minC0 = min(r, min(g, b)), delta0 = maxC0 - minC0;
            if (delta0 == 0) {
                rejGrey++;
                continue;
            }
            if (r < b + MIN_RED_DOMINANCE) {
                rejHue++;
                continue;
            }
            if (g > r + 20) {
                rejHue++;
                continue;
            }
            uint8_t sat0 = (uint8_t)((uint16_t)delta0 * 255 / maxC0);
            if (sat0 < MIN_SATURATION) {
                rejSat++;
                continue;
            }
            int16_t hue0 = _calcHue(r, g, b, maxC0, delta0);
            if (!_hueIsFlame(hue0)) {
                rejHue++;
                continue;
            }
#else
            if (!_isFireColour(r, g, b))
                continue;
#endif
            fire++;

            // ── Update bounding box ────────────────────────
            int pixelNum = (int)(i / 3);
            int px = pixelNum % FRAME_W;
            int py = pixelNum / FRAME_W;
            if (px < _lastBBox.x1)
                _lastBBox.x1 = px;
            if (py < _lastBBox.y1)
                _lastBBox.y1 = py;
            if (px > _lastBBox.x2)
                _lastBBox.x2 = px;
            if (py > _lastBBox.y2)
                _lastBBox.y2 = py;
            _lastBBox.valid = true;
        }

#if FIRE_DEBUG
        Serial.printf("[PIX] bright=%u rejDark=%u rejGrey=%u rejSat=%u rejHue=%u fire=%u | sample=(%u,%u,%u)\n",
                      total - rejDark, rejDark, rejGrey, rejSat, rejHue, fire, sR, sG, sB);
#endif
        return (total == 0) ? 0.0f : (float)fire / (float)total;
    }

    // ── Pixel helpers ────────────────────────────────────────
    static void _setPixel(uint8_t *rgb, int x, int y,
                          uint8_t r, uint8_t g, uint8_t b) {
        if (x < 0 || x >= FRAME_W || y < 0 || y >= FRAME_H)
            return;
        size_t idx = ((size_t)y * FRAME_W + x) * 3;
        rgb[idx] = r;
        rgb[idx + 1] = g;
        rgb[idx + 2] = b;
    }

    static void _drawRect(uint8_t *rgb, const BBox &box,
                          uint8_t r, uint8_t g, uint8_t b,
                          int thickness = 2) {
        // Add a small margin so the box sits just outside the fire cluster
        constexpr int PAD = 6;
        int x1 = max(0, box.x1 - PAD);
        int y1 = max(0, box.y1 - PAD);
        int x2 = min((int)FRAME_W - 1, box.x2 + PAD);
        int y2 = min((int)FRAME_H - 1, box.y2 + PAD);

        for (int t = 0; t < thickness; t++) {
            for (int x = x1 + t; x <= x2 - t; x++) {
                _setPixel(rgb, x, y1 + t, r, g, b); // top
                _setPixel(rgb, x, y2 - t, r, g, b); // bottom
            }
            for (int y = y1 + t; y <= y2 - t; y++) {
                _setPixel(rgb, x1 + t, y, r, g, b); // left
                _setPixel(rgb, x2 - t, y, r, g, b); // right
            }
        }
    }

    // ── Colour math (unchanged) ──────────────────────────────
    static int16_t _calcHue(uint8_t r, uint8_t g, uint8_t b,
                            uint8_t maxC, uint8_t delta) {
        int16_t hue;
        if (maxC == r) {
            hue = 30 * (int16_t)(g - b) / delta;
            if (hue < 0)
                hue += 180;
        } else if (maxC == g) {
            hue = 30 * (int16_t)(b - r) / delta + 60;
        } else {
            hue = 30 * (int16_t)(r - g) / delta + 120;
        }
        return hue;
    }
    static bool _hueIsFlame(int16_t hue) {
        return (hue <= 18) || (hue >= 162) || (hue >= 19 && hue <= 50);
    }
    static bool _isFireColour(uint8_t r, uint8_t g, uint8_t b) {
        uint8_t maxC = max(r, max(g, b)), minC = min(r, min(g, b)), delta = maxC - minC;
        if (maxC < MIN_VALUE)
            return false;
        if (delta == 0)
            return false;
        if (r < b + MIN_RED_DOMINANCE)
            return false;
        if (g > r + 20)
            return false;
        uint8_t sat = (uint8_t)((uint16_t)delta * 255 / maxC);
        if (sat < MIN_SATURATION)
            return false;
        return _hueIsFlame(_calcHue(r, g, b, maxC, delta));
    }

    float _variance() {
        float sum = 0, sumSq = 0;
        for (uint8_t i = 0; i < HIST_LEN; i++) {
            sum += _history[i];
            sumSq += _history[i] * _history[i];
        }
        float mean = sum / HIST_LEN;
        return (sumSq / HIST_LEN) - (mean * mean);
    }
};

// ──────────────────────────────────────────────────────────────
FireDetector fireDetector;
volatile bool fireAlert = false;
volatile float lastFireRatio = 0.0f;

// ── HTTP handlers ──────────────────────────────────────────────
void handleRoot() {
    String page =
        "<html><head><title>ESP32-CAM Fire Monitor</title>"
        "<style>"
        "body{margin:0;background:#111;color:#eee;font-family:sans-serif;}"
        "img{width:100%;display:block;}"
        "#status{padding:10px 16px;font-size:18px;font-weight:bold;"
        "color:#fff;text-align:center;transition:background 0.3s;}"
        "</style></head><body>"
        "<div id='status'>Connecting...</div>"
        "<img src='/stream'>"
        "<script>"
        "function poll(){"
        "  fetch('/status').then(r=>r.json()).then(d=>{"
        "    var s=document.getElementById('status');"
        "    if(d.fire){"
        "      s.style.background='#b91c1c';"
        "      s.textContent='FIRE DETECTED (conf='+d.confidence+')';"
        "    } else {"
        "      s.style.background='#14532d';"
        "      s.textContent='No fire (ratio='+d.ratio.toFixed(4)+')';"
        "    }"
        "  }).catch(()=>{});"
        "}"
        "poll(); setInterval(poll,500);"
        "</script>"
        "</body></html>";
    server.send(200, "text/html", page);
}

void handleStatus() {
    BBox b = fireDetector.lastBBox();
    String json =
        "{\"fire\":" + String(fireAlert ? "true" : "false") +
        ",\"ratio\":" + String(lastFireRatio, 5) +
        ",\"confidence\":" + String(fireDetector.confidence()) +
        ",\"bbox\":{"
        "\"valid\":" +
        String(b.valid ? "true" : "false") +
        ",\"x1\":" + b.x1 +
        ",\"y1\":" + b.y1 +
        ",\"x2\":" + b.x2 +
        ",\"y2\":" + b.y2 +
        "}}";
    server.send(200, "application/json", json);
}

void handleStream() {
    WiFiClient client = server.client();
    client.setNoDelay(true);
    client.print(
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
        "Cache-Control: no-cache\r\n"
        "Connection: close\r\n\r\n");

    while (client.connected()) {
        camera_fb_t *frame = cam.capture();
        if (!frame) {
            yield();
            continue;
        }

        // ── Run detection (decodes JPEG → RGB internally) ──
        bool detected = fireDetector.update(frame->buf, frame->len);
        fireAlert = detected;
        lastFireRatio = fireDetector.lastFireRatio();

        if (detected)
            Serial.println(">>> FIRE DETECTED <<<");

        // ── Choose plain or annotated frame ───────────────
        uint8_t *annotated = nullptr;
        size_t annotatedLen = 0;
        bool useAnnotated = false;

        if (detected) {
            // Draws orange bbox on the decoded RGB buffer, re-encodes JPEG
            useAnnotated = fireDetector.getAnnotatedJpeg(&annotated, &annotatedLen);
        }

        const uint8_t *sendBuf = useAnnotated ? annotated : frame->buf;
        size_t sendLen = useAnnotated ? annotatedLen : frame->len;

        client.print("--frame\r\n");
        client.print("Content-Type: image/jpeg\r\n");
        client.print("Content-Length: " + String(sendLen) + "\r\n\r\n");
        client.write(sendBuf, sendLen);
        client.print("\r\n");

        if (annotated)
            free(annotated); // fmt2jpg heap-allocates; we must free
        cam.returnFrame(frame);
        server.handleClient();
        yield();
    }
}

// ──────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    cam.enableLogging(false);
    cam.init(AI_THINKER(), VGA);

    WiFi.mode(WIFI_STA);
    WiFi.setSleep(false);
    WiFi.begin(ssid, password);
    Serial.print("Connecting");
    while (WiFi.status() != WL_CONNECTED) {
        Serial.print(".");
        delay(300);
    }
    Serial.println("\nConnected: " + WiFi.localIP().toString());

    server.on("/", handleRoot);
    server.on("/stream", handleStream);
    server.on("/status", handleStatus);
    server.begin();
    Serial.println("Ready.");
}

void loop() {
    server.handleClient();
}