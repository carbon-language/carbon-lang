#!/bin/sh

../as/as < ../../test/$1 | ./opt -constprop -dce | ../dis/dis

