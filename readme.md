# e-prop on MoS₂ Memtransistor (device-in-the-loop simülasyon)

e-prop öğrenme kuralının (Bellec 2020) ağırlıklarını, fenomenolojik bir
MoS₂ memtransistor modeline (Sangwan 2018) **pulse-based + gradient-accumulation**
yazma ile gömen bir simülasyon. Görev: pattern generation.

## Veri akışı

```
LSNN forward (network.py)
   └─ online e-prop trace'leri → istenen ΔW = -lr·Σ_t L·ē
        └─ Synapse.update (synapse.py)   [writer: direct / accumulate / verify]
             └─ polarite + n pulse
                  └─ Memtransistor.pulse (memtransistor.py)  → G güncellenir (non-ideal)
       W_rec, W_in = γ·(G - G_ref)   ← bir sonraki trial forward'da okunur
   readout (W_out, b_out): ideal (veya cihazda), doğrudan güncellenir
```

`IdealDevice` = sürekli, gürültüsüz, sınırsız referans (teorik tavan). Loop
her iki cihazda birebir aynı; tek fark cihaz nesnesi → adil ablation.

## Dosyalar
- `config.py` — tüm hiperparametreler (dataclass).
- `device_interface.py` — `ConductanceDevice` ABC + `IdealDevice`.
- `memtransistor.py` — fenomenolojik model: bipolar, durum-bağlı saturasyonlu
  LTP/LTD (Fig. 4c), V_G knob'u, ablatable non-idealiteler.
- `synapse.py` — W↔G köprü + pulse writer (accumulate default).
- `neurons.py` — pseudo-derivative.
- `network.py` — LSNN (LIF+ALIF) + online e-prop (eligibility trace'leri, learning signal, gradyan).
- `task.py` — pattern generation (frozen input → sinüs toplamı).
- `train.py` — episodic e-prop eğitim döngüsü.
- `run_experiment.py` — ideal vs memtransistor + figür.

## Çalıştırma
```bash
pip install torch matplotlib
python run_experiment.py        # learning_curves.png üretir
```
Apple Silicon: `config.py` içinde `TrainConfig.torch_device = "mps"`.

## Planlanan deneyler (config flip)
- **V_G süpürmesi (headline):** `DeviceConfig.V_G` ∈ {-30,0,20,40} → dinamik aralık/granularite vs doğruluk.
- **Writer karşılaştırması:** `SynapseConfig.writer` = "direct" | "accumulate" | "verify".
- **Non-idealite ablation:** `DeviceConfig.enable_{nonlinearity,asymmetry,c2c,d2d}` tek tek kapat → maliyet tablosu.
- **e-prop varyantı:** `TrainConfig.eprop_variant` = "symmetric" | "random" (weight-transport'suz, donanım-akla-yatkın).
- **Readout cihazda:** `TrainConfig.readout_on_device = True` (henüz bağlanmadı; sonraki adım).
- **Nöron:** `NeuronConfig.adaptive_frac` = 0 (saf LIF) ↔ 0.4 (LSNN).

## Makalede AÇIKÇA deklare edilmesi gereken modelleme varsayımları
1. **Episodic e-prop:** gradyan trial boyunca t üzerinden toplanıp trial sonunda
   yazılıyor → gerçek per-timestep online değil. Bu, gradient-accumulation yazma
   şemasının doğal sonucu olan tasarım tercihi olarak konumlandırılmalı.
2. **Fenomenolojik cihaz:** tam fizik (Sangwan denk. 14, 18-19) değil; LTP/LTD
   pulse-yanıtı Fig. 4c'ye fit. Sınır-yakını adım sönümü, pencere fonksiyonunun
   (denk. 17) özünü taşıyor.
3. **V_G haritalaması** (`gate_gain`) fenomenolojik; switching-oranı eğilimini
   (Fig. 2b) yansıtan monoton bir seçim.
4. Tek-cihaz bipolar işaretli ağırlık varsayımı (differential pair yok).

## Sonraki adımlar
- `memtransistor.py`'yi Fig. 4c LTP/LTD verisine nicel fit (dp, dd, kp, kd, V_G).
- Readout'u cihaza taşıma kolu.
- Gradient-check: küçük LIF ağında e-prop gradyanını autograd/BPTT ile doğrula.