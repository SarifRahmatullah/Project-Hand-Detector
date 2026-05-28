import cv2
import mediapipe as mp
import os
import time
import pygame
import warnings

# Abaikan peringatan sistem
warnings.filterwarnings("ignore")

# Inisialisasi Mediapipe
mp_gambar = mp.solutions.drawing_utils
mp_tangan = mp.solutions.hands

# Inisialisasi pygame mixer dengan fallback aman
try:
    pygame.mixer.pre_init(44100, -16, 2, 512)
    pygame.mixer.init()
    AUDIO_TERSEDIA = True
    print("[INFO] Audio mixer berhasil diinisialisasi.")
except Exception as e:
    AUDIO_TERSEDIA = False
    print(f"[PERINGATAN] Mixer gagal init: {e} — program tetap jalan tanpa audio.")

# ===================================================================
# KONFIGURASI AUDIO & MEDIA
# Tambahkan gestur baru di sini sesuai kebutuhan
# ===================================================================
PETA_MEDIA = {
    # "Halo": {
    #     "audio": "audio/halo_saya.mp3",
    #     "tipe": "video",
    #     "jalur": "halo.mp4"
    # },
    # "Mantap": {
    #     "audio": "audio/mantap.mp3",
    #     "tipe": "image",
    #     "jalur": "jempol.jpg"
    # },
    # "Nama saya Sarif Rahmatullah": {
    #     "audio": "audio/perkenalan.mp3",
    #     "tipe": "image",
    #     "jalur": "semangat.png"
    # },
    "Terima Kasih": {
        "audio": "Mania.mp3",
        "tipe": "video",
        "jalur": "kicau_mania.mp4"
    }
}

# ===================================================================
# FUNGSI: Memutar & menghentikan audio
# ===================================================================
def putar_audio(jalur_file):
    """Mulai memutar audio. Non-blocking — tidak menunggu selesai."""
    if not AUDIO_TERSEDIA:
        return
    try:
        if not os.path.exists(jalur_file):
            print(f"[PERINGATAN] File audio tidak ditemukan: '{jalur_file}'")
            return
        pygame.mixer.music.load(jalur_file)
        pygame.mixer.music.play(-1)  # -1 = loop terus sampai di-stop manual
    except Exception as e:
        print(f"[ERROR] Gagal memutar audio: {e}")

def stop_audio():
    """Hentikan audio yang sedang diputar."""
    if not AUDIO_TERSEDIA:
        return
    try:
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
    except Exception as e:
        print(f"[ERROR] Gagal menghentikan audio: {e}")


# ===================================================================
# FUNGSI: Deteksi gestur tangan
# ===================================================================
def deteksi_gestur(titik_tangan):
    """
    Mendeteksi gestur dari landmark tangan MediaPipe.
    Landmark index:
      - Ujung  : 4 (jempol), 8 (telunjuk), 12 (tengah), 16 (manis), 20 (kelingking)
      - Pangkal: 2 (jempol/CMC), 5 (telunjuk), 9 (tengah), 13 (manis), 17 (kelingking)
    Catatan: Sumbu Y mengecil ke atas layar (y kecil = posisi tinggi).
    """
    lm = titik_tangan.landmark

    ujung_jempol    = lm[4].y
    ujung_telunjuk  = lm[8].y
    ujung_tengah    = lm[12].y
    ujung_manis     = lm[16].y
    ujung_kelingking = lm[20].y

    # ✅ PERBAIKAN #1: Gunakan index pangkal yang benar
    pangkal_jempol    = lm[2].y   # CMC joint (index 2)
    pangkal_telunjuk  = lm[5].y
    pangkal_tengah    = lm[9].y
    pangkal_manis     = lm[13].y
    pangkal_kelingking = lm[17].y

    # Shorthand: jari "tegak" = ujung lebih tinggi (y lebih kecil) dari pangkal
    telunjuk_tegak   = ujung_telunjuk  < pangkal_telunjuk
    tengah_tegak     = ujung_tengah    < pangkal_tengah
    manis_tegak      = ujung_manis     < pangkal_manis
    kelingking_tegak = ujung_kelingking < pangkal_kelingking
    jempol_tegak     = ujung_jempol    < pangkal_jempol

    # 1. Halo — Semua jari kecuali jempol tegak (buka telapak)
    #    ✅ PERBAIKAN #5: Ditambah syarat kelingking tegak agar tidak bentrok
    if telunjuk_tegak and tengah_tegak and manis_tegak and kelingking_tegak:
        return "Halo"

    # 2. Mantap — Hanya jempol yang tegak (thumbs up)
    if jempol_tegak and not telunjuk_tegak and not tengah_tegak and not manis_tegak:
        return "Mantap"

    # 3. Nama saya — Hanya telunjuk tegak, jari lain menekuk
    if telunjuk_tegak and not tengah_tegak and not manis_tegak and not kelingking_tegak:
        return "Nama saya Sarif Rahmatullah"

    # 4. Terima Kasih — Telunjuk & tengah tegak, manis & kelingking menekuk
    #    ✅ PERBAIKAN #5: Ditambah syarat kelingking TIDAK tegak agar beda dari "Halo"
    if telunjuk_tegak and tengah_tegak and not manis_tegak and not kelingking_tegak:
        return "Terima Kasih"

    return None


# ===================================================================
# PROGRAM UTAMA
# ===================================================================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Kamera tidak bisa dibuka! Coba ganti index: cv2.VideoCapture(1)")
    exit(1)

print("[INFO] Kamera berhasil dibuka. Tekan ESC untuk keluar.")

video_saat_ini        = None
gestur_aktif          = None
gestur_terakhir_suara = None
waktu_suara_terakhir  = 0
audio_aktif           = False   # True = audio sedang diputar untuk gestur aktif

COOLDOWN_AUDIO = 2.0   # detik jeda minimal antar pemutaran audio
COOLDOWN_RESET = 1.5   # detik sebelum memori gestur direset saat tangan hilang
waktu_tangan_hilang = 0

# ✅ PERBAIKAN #4: Bungkus seluruh loop dengan try/finally agar resource selalu dilepas
try:
    with mp_tangan.Hands(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as tangan:

        while True:
            berhasil, frame = cap.read()
            if not berhasil:
                print("[ERROR] Tidak bisa membaca frame dari kamera.")
                break

            frame = cv2.flip(frame, 1)
            tinggi, lebar, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hasil = tangan.process(rgb_frame)

            gestur = None

            # --- DETEKSI TANGAN ---
            if hasil.multi_hand_landmarks:
                for landmark_tangan in hasil.multi_hand_landmarks:
                    mp_gambar.draw_landmarks(
                        frame, landmark_tangan, mp_tangan.HAND_CONNECTIONS
                    )
                    gestur = deteksi_gestur(landmark_tangan)

                # --- LOGIKA UTAMA (SUARA & MEDIA) ---
                if gestur and gestur in PETA_MEDIA:
                    media = PETA_MEDIA[gestur]

                    # 1. Logika Suara — putar saat gestur baru, loop sampai hilang
                    if gestur != gestur_terakhir_suara:
                        jalur_audio = media.get("audio")
                        if jalur_audio:
                            putar_audio(jalur_audio)
                            audio_aktif = True
                        gestur_terakhir_suara = gestur
                        waktu_suara_terakhir  = time.time()

                    # 2. Logika Tampilan Gambar
                    if media["tipe"] == "image":
                        gambar_overlay = cv2.imread(media["jalur"])
                        if gambar_overlay is not None:
                            gambar_overlay = cv2.resize(gambar_overlay, (200, 200))
                            frame[50:250, lebar - 250:lebar - 50] = gambar_overlay
                        else:
                            print(f"[PERINGATAN] Gambar tidak ditemukan: {media['jalur']}")

                    # 3. Logika Tampilan Video
                    elif media["tipe"] == "video":
                        if gestur_aktif != gestur:
                            if video_saat_ini:
                                video_saat_ini.release()
                            video_saat_ini = cv2.VideoCapture(media["jalur"])
                            gestur_aktif = gestur

                        v_berhasil, v_frame = video_saat_ini.read()
                        if v_berhasil:
                            frame_video = cv2.resize(v_frame, (200, 200))
                            frame[50:250, lebar - 250:lebar - 50] = frame_video
                        else:
                            # Loop video dari awal
                            video_saat_ini.set(cv2.CAP_PROP_POS_FRAMES, 0)

                    # Reset timer hilang karena tangan terdeteksi
                    waktu_tangan_hilang = 0

                else:
                    # Gestur tidak dikenali / tidak ada di PETA_MEDIA — stop audio
                    stop_audio()
                    audio_aktif = False
                    gestur_terakhir_suara = None
                    gestur_aktif = None
                    if video_saat_ini:
                        video_saat_ini.release()
                        video_saat_ini = None

            else:
                # Tangan hilang dari kamera — langsung stop audio
                stop_audio()
                audio_aktif = False

                if waktu_tangan_hilang == 0:
                    waktu_tangan_hilang = time.time()

                if time.time() - waktu_tangan_hilang > COOLDOWN_RESET:
                    gestur_terakhir_suara = None

                gestur_aktif = None
                if video_saat_ini:
                    video_saat_ini.release()
                    video_saat_ini = None

            # --- TAMPILAN HUD ---
            teks_gestur = gestur if gestur else "-"
            cv2.putText(
                frame,
                f"Gestur: {teks_gestur}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2
            )

            cv2.imshow("Sistem Deteksi Gestur - Sarif Rahmatullah", frame)

            # Tekan ESC untuk keluar
            if cv2.waitKey(1) & 0xFF == 27:
                break

finally:
    print("\n[INFO] Menutup program...")
#    cap.release()
#    if video_saat_ini:
#        video_saat_ini.release()
#    if AUDIO_TERSEDIA:
#        pygame.mixer.quit()
#    cv2.destroyAllWindows()
#    print("[INFO] Program selesai.")