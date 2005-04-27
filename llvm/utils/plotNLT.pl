#!/usr/bin/perl
#takes a test and a program from a dp and produces a gnuplot script
#use like perl plotNLT.pl password Programs/MultiSource/Benchmarks/ASCI_Purple/SMG2000/smg2000 llc

use DBI;

# database information
$db="llvmalpha";
$host="localhost";
$userid="llvmdbuser";
$passwd=shift @ARGV;
$connectionInfo="dbi:mysql:$db;$host";

# make connection to database
$dbh = DBI->connect($connectionInfo,$userid,$passwd) or die DBI->errstr;

$prog = shift @ARGV;
$test = shift @ARGV;

print "set xdata time\n";
print 'set timefmt "%Y-%m-%d"';
print "\nplot '-' using 1:2 with lines \n";

$query = "Select RUN, VALUE from Tests where TEST = '$test' AND NAME = '$prog' ORDER BY RUN";
#print $query;

my $sth = $dbh->prepare( $query) || die "Can't prepare statement: $DBI::errstr";;

my $rc = $sth->execute or die DBI->errstr;

while(($da,$v) = $sth->fetchrow_array)
{
  print "$da $v\n";
}

print "e\n";

# disconnect from database
$dbh->disconnect;
