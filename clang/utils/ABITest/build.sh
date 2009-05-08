#!/bin/sh

set -eu

if [ $# != 1 ]; then
    echo "usage: $0 <num-tests>"
    exit 1
fi

CPUS=$(sysctl -n hw.ncpu)
make -j $CPUS \
  $(for i in $(zseq 0 $1); do echo test.$i.report; done) -k
