#!/bin/bash

set -e # fail on any error

OS=`uname`
CXX=$1
CC=$2
FILE_CHECK=$3
CXXFLAGS="-mno-omit-leaf-frame-pointer -fno-omit-frame-pointer -fno-optimize-sibling-calls -g"
SYMBOLIZER=../scripts/asan_symbolize.py
TMP_ASAN_REPORT=asan_report.tmp

run_program() {
  ./$1 2>&1 | $SYMBOLIZER 2> /dev/null | c++filt > $TMP_ASAN_REPORT
}

# check_program exe_file source_file check_prefix
check_program() {
  run_program $1
  $FILE_CHECK $2 --check-prefix=$3 < $TMP_ASAN_REPORT
  rm -f $TMP_ASAN_REPORT
}

C_TEST=use-after-free
echo "Sanity checking a test in pure C"
$CC -g -faddress-sanitizer -O2 $C_TEST.c
check_program a.out $C_TEST.c CHECK
rm ./a.out

echo "Sanity checking a test in pure C with -pie"
$CC -g -faddress-sanitizer -O2 $C_TEST.c -pie
check_program a.out $C_TEST.c CHECK
rm ./a.out

echo "Testing sleep_before_dying"
$CC -g -faddress-sanitizer -O2 $C_TEST.c
export ASAN_OPTIONS="sleep_before_dying=1"
check_program a.out $C_TEST.c CHECKSLEEP
export ASAN_OPTIONS=""
rm ./a.out

for t in  *.cc; do
  for b in 32 64; do
    for O in 0 1 2 3; do
      c=`basename $t .cc`
      if [[ "$c" == *"-so" ]]
      then
        continue
      fi
      c_so=$c-so
      exe=$c.$b.O$O
      so=$c.$b.O$O-so.so
      echo testing $exe
      build_command="$CXX $CXXFLAGS -m$b -faddress-sanitizer -O$O $c.cc -o $exe"
      [ "$DEBUG" == "1" ] && echo $build_command
      $build_command
      [ -e "$c_so.cc" ] && $CXX $CXXFLAGS -m$b -faddress-sanitizer -O$O $c_so.cc -fPIC -shared -o $so
      run_program $exe
      # Check common expected lines for OS.
      $FILE_CHECK $c.cc --check-prefix="Check-Common" < $TMP_ASAN_REPORT
      # Check OS-specific lines.
      if [ `grep -c "Check-$OS" $c.cc` -gt 0 ]
      then
        $FILE_CHECK $c.cc --check-prefix="Check-$OS" < $TMP_ASAN_REPORT
      fi
      rm ./$exe
      rm ./$TMP_ASAN_REPORT
      [ -e "$so" ] && rm ./$so
    done
  done
done

exit 0
