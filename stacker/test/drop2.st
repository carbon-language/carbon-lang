#
# DROP2 test
#
FORWARD success;
FORWARD failure;
: step2 0 = IF success ELSE failure ENDIF ;
: MAIN 0 1 2 3 DROP2 1 = IF step2 ELSE failure ENDIF ;
