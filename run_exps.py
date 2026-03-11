import os
from multiprocessing import Pool


if __name__ == "__main__":
    pool = Pool(processes=1)  # 进程池

    run_file = "train.py"
    train_data_path1 = "/home/node5/AD/datasets/mvtec3d-multiview_672to224"
    train_data_path2 = "/home/node5/AD/datasets/eyecandies-multiview_672to224"
    save_path = "./exps_compare/mvtec"

    dataset1 = "mvtec3d"
    dataset2 = "eyecandies"
    gpu_id = 1

    sh = f'CUDA_VISIBLE_DEVICES={gpu_id} python {run_file} --save_path {save_path} --train_data_path {train_data_path2} --dataset {dataset2} \
            --epoch 3'
    print(f'exec {sh}')
    # pool.apply_async(os.system, (sh,))

    for epoch_num in [3]:
        checkpoint = './exps_compare/epoch_{}.pth'.format(epoch_num)
        sh = f'CUDA_VISIBLE_DEVICES={gpu_id} python ./test.py --save_path ./result_compare/ \
            --checkpoint_path {checkpoint} --data_path {train_data_path1} --dataset {dataset1}'
        print(f'exec {sh}')
        pool.apply_async(os.system, (sh,))

    pool.close()
    pool.join()








