#!/bin/sh

../as/as < ../../test/$1 | ./opt -strip | dis
