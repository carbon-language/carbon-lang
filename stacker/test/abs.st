#
# ABS test
#
FORWARD success;
FORWARD failure;
: MAIN -23 ABS 23 = IF success ELSE failure ENDIF ;
