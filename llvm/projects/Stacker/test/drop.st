#
# DROP test
#
FORWARD success;
FORWARD failure;
: MAIN 1 2 DROP 1 = IF success ELSE failure ENDIF ;
