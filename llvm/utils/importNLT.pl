#!/usr/bin/perl
#take the output of parseNLT.pl and load it into a database

use DBI;

# database information
$db="llvmalpha";
$host="narya.lenharth.org";
$userid="llvmdbuser";
$passwd=""; #removed for obvious reasons
$connectionInfo="dbi:mysql:$db;$host";

# make connection to database
$dbh = DBI->connect($connectionInfo,$userid,$passwd) or die DBI->errstr;

while($d = <>)
{
    if (18 == split / /, $d)
    {
	($day, $mon, $year, $prog, $gccas, $bc, $llc-compile, $llc-beta-compile, $jit-compile,
	 $mc, $gcc, $cbe, $llc, $llc-beta, $jit, $foo1, $foo2, $foo3) = split / /, $d;
	print ".";
    }
}
# disconnect from database
$dbh->disconnect
