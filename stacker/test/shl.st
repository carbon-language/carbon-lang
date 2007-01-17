#
# SHL test
#
FORWARD success;
FORWARD failure;
: show_failure >d SPACE failure ;
: MAIN 64 3 << 512 = IF success ELSE show_failure ENDIF ;
