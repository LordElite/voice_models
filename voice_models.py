import os
import torch
from melo.api import TTS
from openvoice import se_extractor
from openvoice.api import ToneColorConverter

# ---------------------
# CONFIGURACIÓN GLOBAL
# ---------------------
device = "cuda:0" if torch.cuda.is_available() else "cpu"

ckpt_converter = 'checkpoints_v2/converter'
output_dir = 'outputs_v2'
os.makedirs(output_dir, exist_ok=True)

# Cargar conversor de tono una sola vez
tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')


# Inicializar el modelo TTS (puedes ajustar el idioma)




# ---------------------
# FUNCIÓN PRINCIPAL
# ---------------------
def text_to_cloned_speech(text: str, language = 'EN_NEWEST',voice_model = 'aylan', genre = '', subgenre = '', output_name: str = "output.wav") -> str:
    """
    Convierte texto a voz clonada y guarda el archivo.
    Devuelve la ruta del archivo generado.
    """
    
    #modelo de idioma
    tts_model = TTS(language=language, device=device)
    speaker_ids = tts_model.hps.data.spk2id
    
    #modelo de voz
    if subgenre == '':
        target_se = torch.load(f"embeddings/{genre}_{voice_model}.pth", map_location=device)
    else:
        target_se = torch.load(f"embeddings/{genre}_{subgenre}_{voice_model}.pth", map_location=device)
        
    
    
    # Usamos la primera voz del modelo base
    speaker_key = list(speaker_ids.keys())[0]
    speaker_id = speaker_ids[speaker_key]

    # Generamos audio base
    src_path = f'{output_dir}/tmp.wav'
    tts_model.tts_to_file(text, speaker_id, src_path, speed=1.0)

    # Aplicamos clonación de voz
    save_path = f'{output_dir}/{output_name}'
    source_se = torch.load(
        f'checkpoints_v2/base_speakers/ses/{speaker_key.lower().replace("_", "-")}.pth',
        map_location=device
    )
    tone_color_converter.convert(
        audio_src_path=src_path,
        src_se=source_se,
        tgt_se=target_se,
        output_path=save_path
    )

    return save_path

