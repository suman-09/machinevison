#include "esp_camera.h"
#include <WiFi.h>
#include <ArduinoWebsockets.h>
#include "esp_timer.h"
#include "img_converters.h"
#include "fb_gfx.h"
#include "soc/soc.h"             // Disable brownout detector
#include "soc/rtc_cntl_reg.h"
#include "driver/gpio.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include <ESP32Servo.h>

// Hardware Configuration
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22
#define FLASH_LED_PIN      4
#define SERVO_PIN         13

// Network Configuration
const char* WIFI_SSID = "Pixel_7090";
const char* WIFI_PASS = "arnavgupta";
const char* WS_URL = "ws://10.24.12.98:3000/ws";

Servo neck_motor;
QueueHandle_t servoQueue;

// Servo Task
void servoTask(void *pvParameters) {
  int angle;
  while (true) {
    if (xQueueReceive(servoQueue, &angle, portMAX_DELAY) == pdTRUE) {
      angle = constrain(angle, 0, 180);
      neck_motor.write(angle);
      digitalWrite(FLASH_LED_PIN, HIGH);
      delay(1000);
      digitalWrite(FLASH_LED_PIN, LOW);
      Serial.printf("Servo moved to %d degrees!\n", angle);
    }
  }
}

// Global Objects
using namespace websockets;
WebsocketsClient client;
QueueHandle_t frameQueue = NULL;
bool isConnected = false;

// Camera Task (Core 0)
void cameraTask(void *pvParameters) {
  while (1) {
    camera_fb_t *frame = esp_camera_fb_get();
    if (!frame) {
      Serial.println("Camera capture failed");
      continue;
    }

    if (xQueueSend(frameQueue, &frame, 0) != pdTRUE) {
      esp_camera_fb_return(frame);
      Serial.println("Queue full - frame discarded");
    }

    vTaskDelay(pdMS_TO_TICKS(50));
  }
}

// Network Task (Core 1)
void networkTask(void *pvParameters) {
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) {
    vTaskDelay(pdMS_TO_TICKS(500));
  }
  Serial.println("WiFi connected");

  // WebSocket Handler
  client.onMessage([](WebsocketsMessage msg) {
    String data = msg.data();
    Serial.printf("Got message: %s\n", data.c_str());

    if (data.startsWith("servo:")) {
      int angle = data.substring(6).toInt();
      xQueueSend(servoQueue, &angle, 0);
    }
  });

  while (1) {
    if (!isConnected) {
      if (client.connect(WS_URL)) {
        isConnected = true;
        client.send("ESP32-CAM");
        Serial.println("WebSocket connected");
      }
    }

    if (isConnected) {
      client.poll();
      camera_fb_t *frame = NULL;

      if (xQueueReceive(frameQueue, &frame, pdMS_TO_TICKS(100)) == pdTRUE) {
        bool success = client.sendBinary((const char *)frame->buf, frame->len);
        if (!success) {
          Serial.println("Send failed - disconnecting");
          client.close();
          isConnected = false;
        }
        esp_camera_fb_return(frame);
      }
    }
    vTaskDelay(1);
  }
}

void setup() {
  // Disable brownout detector
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);
  Serial.begin(115200);

  // Servo Setup
  neck_motor.setPeriodHertz(50);
  neck_motor.attach(SERVO_PIN, 500, 2400);
  neck_motor.write(90); // Center position
  servoQueue = xQueueCreate(5, sizeof(int));

  // Camera Configuration
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_VGA;
  config.jpeg_quality = 15;
  config.fb_count = 2;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x", err);
    ESP.restart();
  }

  frameQueue = xQueueCreate(4, sizeof(camera_fb_t *));

  // Tasks
  xTaskCreatePinnedToCore(cameraTask, "Camera", 4096, NULL, 2, NULL, 0);
  xTaskCreatePinnedToCore(networkTask, "Network", 8192, NULL, 1, NULL, 1);
  xTaskCreatePinnedToCore(servoTask, "Servo", 2048, NULL, 1, NULL, 1);

  // Flash LED
  pinMode(FLASH_LED_PIN, OUTPUT);
  digitalWrite(FLASH_LED_PIN, LOW);
}

void loop() {
  vTaskDelete(NULL);
}
