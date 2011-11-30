#!/bin/bash

OS=`uname`
CXX=$1
CXXFLAGS="-mno-omit-leaf-frame-pointer"
SYMBOLIZER=../scripts/asan_symbolize.py

for t in  *.tmpl; do
  for b in 32 64; do
    for O in 1 2 3; do
      # TODO: reinstate -O0, if that's really needed.
      c=`basename $t .tmpl`
      c_so=$c-so
      exe=$c.$b.O$O
      so=$c_so.$b.O$O.so
      $CXX $CXXFLAGS -g -m$b -faddress-sanitizer -O$O $c.cc -o $exe
      [ -e "$c_so.cc" ] && $CXX $CXXFLAGS -g -m$b -faddress-sanitizer -O$O $c_so.cc -fPIC -shared -o $so
      # If there's an OS-specific template, use it.
      # Please minimize the use of OS-specific templates.
      if [ -e "$t.$OS" ]
      then
        actual_t="$t.$OS"
      else
        actual_t="$t"
      fi
      ./$exe 2>&1 | $SYMBOLIZER 2> /dev/null | c++filt | ./match_output.py $actual_t || exit 1
      echo $exe
      rm ./$exe
      [ -e "$so" ] && rm ./$so
    done
  done
done
exit 0
