#!/usr/bin/fish
micromamba activate pytorcher
# python3 main_runner.py --config "fish_test_proxy" --name "first_run_test_proxy"
# python3 main_runner.py --config "fish_test_proxy_small" --name "first_run_test_proxy_0.1"
#python3 main_runner.py --config "fish_test_proxy_no_train" --name "first_run_test_proxy_no_train"
# python3 main_runner.py --config "asl_test_proxy_train" --name "asl_test_proxy_train"
python3 main_runner.py --config "asl_test_proxy_train_50" --name "asl_test_proxy_train_50"
# python3 main_runner.py --config "asl_test_no_proxy_train" --name "asl_test_no_proxy_train"
# python3 main_runner.py --config "fish_test_no_proxy" --name "first_run_test_noproxy"
