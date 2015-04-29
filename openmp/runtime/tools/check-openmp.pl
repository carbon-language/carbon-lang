#!/usr/bin/perl

use strict;
use warnings;

use FindBin;
use lib "$FindBin::Bin/lib";

# LIBOMP modules.
use Build;
use LibOMP;
use Platform ":vars";
use Uname;
use tools;

my $root_dir  = $ENV{ LIBOMP_WORK };
print join('', $root_dir, "/", "exports", "/", $target_platform, "/", "lib");

