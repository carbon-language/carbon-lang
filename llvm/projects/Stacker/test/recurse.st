#
# RECURSE test
#
FORWARD success;
FORWARD failure;
: recurser 100 = IF 1 + RECURSE;
: MAIN 1 recurser success ;
