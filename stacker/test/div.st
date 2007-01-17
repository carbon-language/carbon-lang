#
# DIV test
#
FORWARD success;
FORWARD failure;
: MAIN 7 49 / 7 = IF success ELSE failure ENDIF ;
