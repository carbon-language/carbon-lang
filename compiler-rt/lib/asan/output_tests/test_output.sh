#!/bin/bash

set -e # fail on any error

OS=`uname`
CXX=$1
CC=$2
CXXFLAGS="-mno-omit-leaf-frame-pointer -fno-omit-frame-pointer -fno-optimize-sibling-calls"
SYMBOLIZER=../scripts/asan_symbolize.py
FILE_CHECK=../../../../../build/Release+Asserts/bin/FileCheck

# check_program exe_file src_file [check_prefix]
check_program() {
  exe=$1
  src=$2
  prefix="CHECK"
  if [ "z$3" != "z" ] ; then
    prefix=$3
  fi
  ./$exe 2>&1 | $SYMBOLIZER 2> /dev/null | c++filt | \
        $FILE_CHECK $src --check-prefix=$prefix
}

C_TEST=use-after-free
echo "Sanity checking a test in pure C"
$CC -g -faddress-sanitizer -O2 $C_TEST.c
check_program a.out $C_TEST.c
rm ./a.out

echo "Sanity checking a test in pure C with -pie"
$CC -g -faddress-sanitizer -O2 $C_TEST.c -pie
check_program a.out $C_TEST.c
rm ./a.out

echo "Testing sleep_before_dying"
$CC -g -faddress-sanitizer -O2 $C_TEST.c
export ASAN_OPTIONS="sleep_before_dying=1"
check_program a.out $C_TEST.c CHECKSLEEP
export ASAN_OPTIONS=""
rm ./a.out

for t in  *.tmpl; do
  for b in 32 64; do
    for O in 0 1 2 3; do
      c=`basename $t .tmpl`
      c_so=$c-so
      exe=$c.$b.O$O
      so=$c.$b.O$O-so.so
      echo testing $exe
      build_command="$CXX $CXXFLAGS -g -m$b -faddress-sanitizer -O$O $c.cc -o $exe"
      [ "$DEBUG" == "1" ] && echo $build_command
      $build_command
      [ -e "$c_so.cc" ] && $CXX $CXXFLAGS -g -m$b -faddress-sanitizer -O$O $c_so.cc -fPIC -shared -o $so
      # If there's an OS-specific template, use it.
      # Please minimize the use of OS-specific templates.
      if [ -e "$t.$OS" ]
      then
        actual_t="$t.$OS"
      else
        actual_t="$t"
      fi
      ./$exe 2>&1 | $SYMBOLIZER 2> /dev/null | c++filt | ./match_output.py $actual_t
      rm ./$exe
      [ -e "$so" ] && rm ./$so
    done
  done
done

exit 0
