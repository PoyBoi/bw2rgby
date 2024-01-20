def colorize(
    file_loc: str,
    is_img: bool = True,
    method: int = 1,
    show_graph: bool = False,
    save_image: bool = False,
    use_gpu: bool = False
):
    # method == 0: ECCV; method == 1: SIGGRAPH (default)
    import os
    import torch
    import argparse
    import matplotlib.pyplot as plt

#   from colorizers import *
#   import colorizers

    from colorizers import eccv16, siggraph17, load_img, preprocess_img, postprocess_tens

    # download Image if it's a link
    # file_loc = ""
    if "https" in file_loc or ".com" in file_loc:
        # get current directory, and add / to the end of it
        cwd = os.popen("pwd").read().split()[0] + r"/"

        # get list of files
        ls = os.popen("ls").read().split()

        # print(cwd)
        name_check = file_loc.split("/")[-1]
        if name_check in ls:
            print("Image already exists, using locally stored version")
        else:
            print("Image not found, downloading")
            os.system("wget -P {} {}".format(cwd, file_loc))
        file_loc = cwd + name_check
    # use local image
    else:
        None

    # print(file_loc)

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--img_path', type=str,
                        default='/content/download_3.jpg')
    parser.add_argument('--use_gpu', action='store_true',
                        help='whether to use GPU')
    parser.add_argument('-o', '--save_prefix', type=str, default='saved',
                        help='will save into this file with {eccv16.png, siggraph17.png} suffixes')
    opt = parser.parse_args(args=["-i", file_loc, use_gpu])

    # parser = argparse.ArgumentParser()
    # parser.add_argument('-i', '--img_path', type=str,
    #                     default='imgs/ansel_adams3.jpg')
    # parser.add_argument('--use_gpu', action='store_true',
    #                     help='whether to use GPU')
    # parser.add_argument('-o', '--save_prefix', type=str, default='saved',
    #                     help='will save into this file with {eccv16.png, siggraph17.png} suffixes')
    # opt = parser.parse_args(args=[])

    # load colorizers
    colorizer_eccv16 = eccv16(pretrained=True).eval()
    colorizer_siggraph17 = siggraph17(pretrained=True).eval()
    if (opt.use_gpu):
        colorizer_eccv16.cuda()
        colorizer_siggraph17.cuda()

    # default size to process images is 256x256
    # grab L channel in both original ("orig") and resized ("rs") resolutions
    img = load_img(opt.img_path)
    (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))
    if (opt.use_gpu):
        tens_l_rs = tens_l_rs.cuda()

    # colorizer outputs 256x256 ab map
    # resize and concatenate to original L channel
    img_bw = postprocess_tens(tens_l_orig, torch.cat(
        (0*tens_l_orig, 0*tens_l_orig), dim=1))
    out_img_eccv16 = postprocess_tens(
        tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
    out_img_siggraph17 = postprocess_tens(
        tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())

    plt.imsave('%s_eccv16.png' % opt.save_prefix, out_img_eccv16)
    plt.imsave('%s_siggraph17.png' % opt.save_prefix, out_img_siggraph17)

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.imshow(img)
    plt.title('Original')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(img_bw)
    plt.title('Input')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(out_img_eccv16)
    plt.title('Output (ECCV 16)')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(out_img_siggraph17)
    plt.title('Output (SIGGRAPH 17)')
    plt.axis('off')
    plt.show()
