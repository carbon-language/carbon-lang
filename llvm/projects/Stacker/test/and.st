#
# AND test
#
FORWARD success;
FORWARD failure;
: step2 7 15 AND 7 = IF success ELSE failure ENDIF ;
: MAIN 8 16 AND 0 = IF step2 ELSE failure ENDIF ;
