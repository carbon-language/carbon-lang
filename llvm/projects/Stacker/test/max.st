#
# MAX test
#
FORWARD success;
FORWARD failure;
: step2 2 1 MAX 2 = IF success ELSE failure ENDIF ;
: MAIN 1 2 MAX 2 = IF step2 ELSE failure ENDIF ;
