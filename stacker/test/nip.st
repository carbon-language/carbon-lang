#
# NIP test
#
FORWARD success;
FORWARD failure;
: MAIN 1 2 NIP 2 = IF success ELSE failure ENDIF ;
