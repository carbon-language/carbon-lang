#!/bin/sh

../as/as < ../../test/$1 | ./opt -inline -constprop -dce | dis
