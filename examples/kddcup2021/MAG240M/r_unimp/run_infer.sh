CUDA_VISIBLE_DEVICES=2 python r_unimp_multi_gpu_infer.py --conf configs/r_unimp_m2v_64_0.yaml --do_eval

sleep 3

python  check_cv_get_test_result.py --conf configs/r_unimp_m2v_64_0.yaml
sleep 3

CUDA_VISIBLE_DEVICES=2 python r_unimp_multi_gpu_infer.py --conf configs/r_unimp_m2v_64_1.yaml --do_eval

sleep 3

python  check_cv_get_test_result.py --conf configs/r_unimp_m2v_64_1.yaml

sleep 3

CUDA_VISIBLE_DEVICES=2 python r_unimp_multi_gpu_infer.py --conf configs/r_unimp_m2v_64_2.yaml --do_eval

sleep 3

python  check_cv_get_test_result.py --conf configs/r_unimp_m2v_64_2.yaml

sleep 3

CUDA_VISIBLE_DEVICES=2 python r_unimp_multi_gpu_infer.py --conf configs/r_unimp_m2v_64_3.yaml --do_eval

sleep 3

python  check_cv_get_test_result.py --conf configs/r_unimp_m2v_64_3.yaml

sleep 3
CUDA_VISIBLE_DEVICES=2 python r_unimp_multi_gpu_infer.py --conf configs/r_unimp_m2v_64_4.yaml --do_eval

sleep 3

python  check_cv_get_test_result.py --conf configs/r_unimp_m2v_64_4.yaml

sleep 3

python  check_cv_get_test_result.py --conf configs/r_unimp_m2v_64_4.yaml --eval_all
