python scripts/generate.py cfg/generative/deepsdf_latent_v1.yaml -m train --gen_args "{"gen_from_known_latent": true}" -o out/models/generative/deepsdf_latent_v1/out/train/
python scripts/generate.py cfg/generative/deepsdf_latent_oc_v1.yaml -m train --gen_args "{"gen_from_known_latent": true}" -o out/models/generative/deepsdf_latent_oc_v1/out/train/

python scripts/generate.py cfg/generative/deepsdf_latent_v2.yaml -m train --gen_args "{"gen_from_known_latent": true}" -o out/models/generative/deepsdf_latent_v2/out/train/
python scripts/generate.py cfg/generative/deepsdf_latent_oc_v2.yaml -m train --gen_args "{"gen_from_known_latent": true}" -o out/models/generative/deepsdf_latent_oc_v2/out/train/