#
# ADD test
#
FORWARD success;
FORWARD failure;
: step2 7 93 + 100 = IF success ELSE failure ENDIF ;
: MAIN 93 7 + 100 = IF step2 ELSE failure ENDIF ;
