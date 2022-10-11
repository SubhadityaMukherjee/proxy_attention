#!/usr/bin/fish
micromamba activate pytorcher
# python3 main_runner.py --config "fish_test_proxy" --name "first_run_test_proxy"
python3 main_runner.py --config "fish_test_proxy_small" --name "first_run_test_proxy_0.1"
# python3 main_runner.py --config "fish_test_no_proxy" --name "first_run_test_noproxy"