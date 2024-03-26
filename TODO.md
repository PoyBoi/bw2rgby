Stuff in the same lane:

1. Try solving the issues on the BOPBTL repo
2. Before adding a PR, open a issue, and then see

</hr>

### Code needed to be made:

Code in order:

1. Image restoration with scratches
   1. Making UNet
   2. Inpainting and filling in the mask
2. Coloring/Color Correction
   1. Face Detection/Enhancement
3. Image Quality Improvement
4. GUI using Gradio
5. Making a universal launcher for cli
</hr>

### Order of work to be done

WRT to code in order:

1. Read up from with code, do what you understand
   1. Do so from the code
   2. Try to see if there are any other better methods, if not, use the method they are using
      - ✅[No scratch] |-> `run.py --with_scratch -> test.py --test_mode --Quality_restore`
      - ✅[Scratch] |-> `detection.py --input_size full_size --GPU |-> main(config)`
2. Read up on the code give
   1. Use GFPGAN/RealESRGAN
      - run.py <input folder // output folder>
      - run.py -> `test.py --Scratch_and_Quality_Restore --test_input --test_mask`
      - Face Detection:
        - [Face Detection w/High Res] -> `detect_all_dlib_HR.py --url --save_url`
        - [Face Detection w/o High Res] -> `detect_all_dlib.py --url --save_url`
      - Face Restoration / Enhancement:
        - [w/ High Resolution] -> `checkpoint -> test_face.py --old_face_folder --old_face_label_folder --tensorboard_log --name --load_size 512 --label_nc 18 --no_instance --preprocess_mode resize --batchSize 1 --results_dir --no_parsing_map`
        - [w/o High Resolution] -> `test_face.py -> --old_face_folder --old_face_label_folder --tensorboard_log --name <ckpt> --gpu_ids --load_size 256 --label_nc 18 --no_instance --preprocess_mode resize --batchSize 4 --results_dir --no_parsing_map`
      - Blending the final results:
        - [w/ High Resolution] -> `align_warp_back_multiple_dlib_HR.py --origin_url --replace_url --save_url`
        - [w/o High Resolution] -> `align_warp_back_multiple_dlib.py --origin_url --replace_url --save_url`
3. Already made, implement / extend it's usage
   - run.py --with_scratch --HR
4. Simple enough, make argparse like zB
</hr>

### In order of importance/logic

todo's:

1. Read up on:
   1. ~Deep Latent Space Translation~
   2. ResBlock
   3. Triplet domain translation
2. Training:
   1. Need to find out how they are training it
   2. Find the dataset
   3. Understand the training
3. Imports
   1. ~Synchronized-BatchNorm-PyTorch~ [Calls torch's batch normalization, just synced]
   2. ~Landmark Detection~ [Landmarks in faces, important points, outputs locations, helps to detect things (emotions, eye position, etc)]
   3. The "pretrained" model
   4. ~The requirements~
      - dlib: ML packages, algo's - mainly for c++, ported to python
      - easyDict: Helps to make parsing dictionaries easier, esp JSON's
      - dominate: Creates HTML pages
      - dill: Python pickling library
      - einops: Extends tensor packages with extra math features added to them

### MISC Links:

- [Replicate's website for BOPBTL](https://replicate.com/microsoft/bringing-old-photos-back-to-life), give this a read
