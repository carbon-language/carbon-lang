#!/bin/bash

bazel test :string_literal_test :string_literal_bm --test_output=errors
echo fastbuild
bazel run -c fastbuild :string_literal_bm |& grep BM_
echo opt
bazel run -c opt :string_literal_bm |& grep BM_
