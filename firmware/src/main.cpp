/**
 * Amaç: ESP32 ile TMC2208 sürücüsü üzerinden T8 vidalı mili 5 dakika boyunca 
 * git-gel (ping-pong) hareketi yaptırmak ve her döngüde hızı kademeli artırmak.
 * 
 * Örnek JSON Çıktıları:
 * // Başlangıç:
 * {"event": "startup", "message": "5 dakikalik test basladi", "max_speed": 1000.0}
 * // Yön ve Hız Değişimi:
 * {"event": "direction_change", "new_target_steps": 200000, "new_max_speed": 1500.0, "acceleration": 750.0}
 * // Periyodik Durum Logu:
 * {"event": "status", "elapsed_time_ms": 15000, "current_position": 45320}
 * // Bitiş Logu:
 * {"event": "complete", "message": "300000ms doldu. Motor durduruldu."}
 */

#include <Arduino.h>
#include <AccelStepper.h>

// --- Pin Tanımlamaları (ESP32) ---
const int STEP_PIN = 26; // Sürücü STEP pini
const int DIR_PIN = 27;  // Sürücü DIR pini
const int EN_PIN = 25;   // Sürücü ENABLE pini (TMC2208)

// --- Mekanik Kalibrasyon ---
const long STEPS_PER_MM = 400;             // 1 mm için gereken adım sayısı
const long TRAVEL_DISTANCE_MM = 500;       // Git-gel mesafesi (50 cm)
const long TARGET_POSITION = TRAVEL_DISTANCE_MM * STEPS_PER_MM;

// --- Zaman ve Hız Değişkenleri ---
const unsigned long TEST_DURATION_MS = 300000; // 5 Dakika
unsigned long startTime = 0;
unsigned long lastLogTime = 0;

float currentMaxSpeed = 1000.0;  // Başlangıç hızı (adım/saniye)
float speedIncrement = 500.0;    // Her uç noktada artırılacak hız miktarı
float maxSpeedLimit = 8000.0;    // ESP32 ve Sürücü sınırını aşmamak için tavan hız

// Sürücü nesnesi (Step ve Dir pinleri ile)
AccelStepper stepper(AccelStepper::DRIVER, STEP_PIN, DIR_PIN);

void setup() {
    Serial.begin(115200);

    // Sürücüyü aktif et (TMC2208 için LOW aktiftir)
    pinMode(EN_PIN, OUTPUT);
    digitalWrite(EN_PIN, LOW);

    // Başlangıç parametreleri
    stepper.setMaxSpeed(currentMaxSpeed);
    stepper.setAcceleration(currentMaxSpeed / 2.0); // İvmelenme hızın yarısı
    stepper.moveTo(TARGET_POSITION);

    startTime = millis();

    // Başlangıç JSON Logu
    Serial.print("{\"event\": \"startup\", \"message\": \"5 dakikalik test basladi\", \"max_speed\": ");
    Serial.print(currentMaxSpeed);
    Serial.println("}");
}

void loop() {
    unsigned long currentTime = millis() - startTime;

    // 1. ZAMAN KONTROLÜ (5 Dakika Doldu mu?)
    if (currentTime >= TEST_DURATION_MS) {
        if (stepper.isRunning()) {
            stepper.stop();
            digitalWrite(EN_PIN, HIGH); // Sürücüyü kapat, motorun ısınmasını engelle
            
            Serial.println("{\"event\": \"complete\", \"message\": \"300000ms doldu. Motor durduruldu.\"}");
        }
        return; // Programı sonlandır (loop döner ama içeriğe girmez)
    }

    // 2. GİT-GEL VE HIZLANMA MANTIĞI
    if (stepper.distanceToGo() == 0) {
        // Hızı artır (Tavan hıza ulaşılmadıysa)
        if (currentMaxSpeed < maxSpeedLimit) {
            currentMaxSpeed += speedIncrement;
            stepper.setMaxSpeed(currentMaxSpeed);
            stepper.setAcceleration(currentMaxSpeed / 2.0);
        }

        // Yönü tersine çevir
        long newTarget = (stepper.currentPosition() == TARGET_POSITION) ? 0 : TARGET_POSITION;
        stepper.moveTo(newTarget);

        // Aksiyon JSON Logu
        Serial.print("{\"event\": \"direction_change\", \"new_target_steps\": ");
        Serial.print(newTarget);
        Serial.print(", \"new_max_speed\": ");
        Serial.print(currentMaxSpeed);
        Serial.print(", \"acceleration\": ");
        Serial.print(currentMaxSpeed / 2.0);
        Serial.println("}");
    }

    // Motoru yürüt
    stepper.run();

    // 3. PERİYODİK DURUM LOGU (Her 5 saniyede bir JSON çıktısı üret)
    if (millis() - lastLogTime > 5000) {
        lastLogTime = millis();
        Serial.print("{\"event\": \"status\", \"elapsed_time_ms\": ");
        Serial.print(currentTime);
        Serial.print(", \"current_position\": ");
        Serial.print(stepper.currentPosition());
        Serial.println("}");
    }
}
