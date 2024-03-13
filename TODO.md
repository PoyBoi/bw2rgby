# TODO FILE

</hr>

### Code needed to be made:

code in order:

1. Image restoration with scratches
   1. Making UNet
   2. Inpainting and filling in the mask
2. Coloring/Color Correction
   1. Face Detection/Enhancement
3. GUI using Gradio
4. Making a universal launcher for cli

</hr>

### Order of work to be done

WRT to code in order:

1. Read up from with code, do what you understand
   1. Do so from the code
   2. Try to see if there are any other better methods, if not, use the method they are using
      - [No scratch] |-> run.py --with_scratch -> test.py --test_mode --Quality_restore
      - [Scratch] |-> detection.py
2. Read up on the code give
   1. Use GFPGAN/RealESRGAN
      - run.py <input folder // output folder>
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
