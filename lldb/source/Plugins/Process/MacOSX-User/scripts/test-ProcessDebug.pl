#!/usr/bin/perl

use strict;
use Cwd 'abs_path';
our $home = $ENV{HOME} || die "ERROR: Couldn't deduce your home directory...\n";

our @inc_paths = (
	'./include',	
);

my $inc_paths_added = 0;
foreach my $inc_path (@inc_paths)
{
	if (-e $inc_path)
	{
		push (@INC, abs_path($inc_path));
		$inc_paths_added++;
	} 
}

if ($inc_paths_added == 0)
{
	die "Please compile the Release version of lldb\n";
}

require lldb;

# my $state = lldb::eStateAttaching;

use constant UINT32_MAX => 4294967295;

#----------------------------------------------------------------------
# Interactive Commands
#----------------------------------------------------------------------
our %commands = (
	break => {
		name => 'break',	# in case an alias is used to get to this command
		description => "Sets a breakpoint.",
		usage => ["break ADDR"],
		function => \&command_set_breakpoint,
		runs_target => 0,
	},
	delete => {
		name => 'delete',	# in case an alias is used to get to this command
		description => "Deletes one or more breakpoints by ID.\
If no breakpoint IDs are given all breakpoints will be deleted.\
If one or more IDs are given, only those breakpoints will be deleted.",
		usage => ["delete [ID1 ID2 ...]"],
		function => \&command_clear_breakpoint,
		runs_target => 0,
	},
	continue => {
		name => 'continue',	# in case an alias is used to get to this command
		description => "Continues target execution.",
		usage => ["continue [ADDR]"],
		function => \&command_continue,
		runs_target => 1
	},
	step => {
		name => 'step',	# in case an alias is used to get to this command
		description => "Single steps one instruction.",
		usage => ["step"],
		function => \&command_step,
		runs_target => 1
	},
	info => {
		name => 'info',	# in case an alias is used to get to this command
		description => "Gets info on a variety of things.",
		usage => ["info reg", "info thread", "info threads"],
		function => \&command_info,
		runs_target => 0
	},
	help => {
		name => 'help',	# in case an alias is used to get to this command
		description => "Displays a list of all commands, or help for a specific command.",
		usage => ["help", "help CMD"],
		function => \&command_help,
		runs_target => 0
	}
);

#----------------------------------------------------------------------
# Command aliases
#----------------------------------------------------------------------
our %aliases = (
	b => $commands{break},
	c => $commands{continue},
	s => $commands{step},
	d => $commands{delete},
	h => $commands{help}	
);

our $opt_g = 0;	# Enable verbose debug logging
our $opt_v = 0;	# Verbose mode
my $prev_command_href = undef;
my $stdio = '/dev/stdin';
my $launch = 0;
my @env = ();
my @break_ids;

#----------------------------------------------------------------------
# Given a command string, return the command hash reference for it, or
# undef if it doesn't exist.
#----------------------------------------------------------------------
sub get_command_hash_ref
{
	my $cmd = shift;
	my $cmd_href = undef;
	if (length($cmd) == 0)			{ $cmd_href = $prev_command_href;	}
	elsif (exists $aliases{$cmd})	{ $cmd_href = $aliases{$cmd};		} 
	elsif (exists $commands{$cmd})	{ $cmd_href = $commands{$cmd};		}
	defined $cmd_href and $prev_command_href = $cmd_href;
	return $cmd_href;
}

#----------------------------------------------------------------------
# Set a breakpoint
#----------------------------------------------------------------------
sub command_set_breakpoint
{
	my $pid = shift;
	my $tid = shift;
	$opt_g and print "command_set_breakpoint (pid = $pid, locations = @_)\n";
	foreach my $location (@_)
	{
		my $success = 0;
		my $address = hex($location);
		if ($address != 0)
		{
			my $break_id = lldb::PDBreakpointSet ($pid, $address, 1, 0);
			if ($break_id != $lldb::PD_INVALID_BREAK_ID)
			{
				printf("Breakpoint %i is set.\n", $break_id);
				push(@break_ids, $break_id);
				$success = 1;
			}
		}
		$success or print("error: failed to set breakpoint at $location.\n");
	}
	return 1;
}

#----------------------------------------------------------------------
# Clear a breakpoint
#----------------------------------------------------------------------
sub command_clear_breakpoint
{
	my $pid = shift;
	my $tid = shift;
	if (@_)
	{
		my $break_id;
		my @cleared_break_ids;
		my @new_break_ids;
		$opt_g and print "command_clear_breakpoint (pid = $pid, break_ids = @_)\n";
		foreach $break_id (@_)
		{
			if (lldb::PDBreakpointClear ($pid, $break_id))
			{
				printf("Breakpoint %i has been cleared.\n", $break_id);
				push (@cleared_break_ids, $break_id);
			}
			else
			{
				printf("error: failed to clear breakpoint %i.\n", $break_id);					
			}
		}
		
		foreach my $old_break_id (@break_ids)
		{
			my $found_break_id = 0;
			foreach $break_id (@cleared_break_ids)
			{
				if ($old_break_id == $break_id)
				{
					$found_break_id = 1;
				}
			}
			$found_break_id or push (@new_break_ids, $old_break_id);
		}
		@break_ids = @new_break_ids;
	}
	else
	{
		# Nothing specified, clear all breakpoints
		return command_clear_breakpoint($pid, $tid, @break_ids);
	}
	return 1;
}
#----------------------------------------------------------------------
# Continue program execution
#----------------------------------------------------------------------
sub command_continue
{
	my $pid = shift;
	my $tid = shift;
	$opt_g and print "command_continue (pid = $pid)\n";
	if ($pid != $lldb::PD_INVALID_PROCESS_ID)
	{
		$opt_v and printf("Resuming pid %d...\n", $pid);
		return lldb::PDProcessResume ($pid);
	}
	return 0;
}

sub command_step
{
	my $pid = shift;
	my $tid = shift;
	$opt_g and print "command_step (pid = $pid, tid = $tid)\n";
	if ($pid != $lldb::PD_INVALID_PROCESS_ID)
	{
		$opt_v and printf("Single stepping pid %d tid = %4.4x...\n", $pid, $tid);
		return lldb::PDThreadResume ($pid, $tid, 1);
	}
	return 0;
}

sub command_info
{
	my $pid = shift;
	my $tid = shift;
	$opt_g and print "command_step (pid = $pid, tid = $tid)\n";
	if ($pid != $lldb::PD_INVALID_PROCESS_ID)
	{
		if (@_)
		{
			my $info_cmd = shift;
			if ($info_cmd eq 'reg')
			{
				
			}
			elsif ($info_cmd eq 'thread')
			{
				# info on the current thread
				printf("thread 0x%4.4x %s\n", $tid, lldb::PDThreadGetInfo($pid, $tid));
			}
			elsif ($info_cmd eq 'threads')
			{
				my $num_threads = lldb::PDProcessGetNumThreads( $pid );
				for my $thread_num (1..$num_threads)
				{
					my $curr_tid = lldb::PDProcessGetThreadAtIndex ( $pid, $thread_num - 1 );
					printf("%c%u - thread 0x%4.4x %s\n", $curr_tid == $tid ? '*' : ' ', $thread_num, $curr_tid, lldb::PDThreadGetInfo($pid, $curr_tid));
				}
			}
		}
	}
	return 1;
}
#----------------------------------------------------------------------
# Get help on all commands, or a specific list of commands
#----------------------------------------------------------------------
sub command_help
{
	my $pid = shift;
	my $tid = shift;
	if (@_)
	{
		$opt_g and print "command_continue (pid = $pid, commands = @_)\n";
		foreach my $cmd (@_)
		{
			my $cmd_href = get_command_hash_ref($cmd);
			if ($cmd_href)
			{
				print '#', '-' x 72, "\n# $cmd_href->{name}\n", '#', '-' x 72, "\n";
				my $usage_aref = $cmd_href->{usage};
				if (@{$usage_aref})
				{
					print "  USAGE\n";
					foreach my $usage (@{$usage_aref}) {
						print "    $usage\n";
					}
					print "\n";
				}
				print "  DESCRIPTION\n    $cmd_href->{description}\n\n";
			}
			else
			{
				print "  invalid command: '$cmd'\n\n";
			}
		}
	}
	else
	{
		return command_help($pid, sort keys %commands);
	}
	return 1;
}


#lldb::PDLogSetLogMask ($lldb::PD_LOG_ALL);
#lldb::PDLogSetLogFile ('/dev/stdout');

print "running: ", join(' ', @ARGV), "\n";

my $pid = lldb::PDProcessLaunch ($ARGV[0], \@ARGV, \@env, "i386", '/dev/stdin', '/dev/stdout', '/dev/stderr', $launch, '', 0);
my $pid_state;
while ($pid)
{
	$opt_g and printf("PDProcessWaitForEvents (%d, 0x%4.4x, SET, 1)\n", $pid, $lldb::PD_ALL_EVENTS);
	my $events = lldb::PDProcessWaitForEvents ($pid, $lldb::PD_ALL_EVENTS, 1, 1);
	if ($events)
	{
		$opt_g and printf ("Got event: 0x%8.8x\n", $events);

		if ($events & $lldb::PD_EVENT_IMAGES_CHANGED)
		{
			$opt_g and printf("pid %d images changed...\n", $pid);
		}

		if ($events & $lldb::PD_EVENT_STDIO)
		{
			$opt_g and printf("pid %d has stdio...\n", $pid);
		}

		if ($events & $lldb::PD_EVENT_ASYNC_INTERRUPT)
		{
			$opt_g and printf("pid %d got async interrupt...\n", $pid);
		}

		if ($events & $lldb::PD_EVENT_RUNNING)
		{
			$pid_state = lldb::PDProcessGetState ($pid);
			$opt_v and printf( "pid %d state: %s.\n", $pid, lldb::PDStateAsString ($pid_state) );
		}
		
		if ($events & $lldb::PD_EVENT_STOPPED)
		{
			$pid_state = lldb::PDProcessGetState ($pid);
			$opt_v and printf( "pid %d state: %s.\n", $pid, lldb::PDStateAsString ($pid_state) );

			if ($pid_state == $lldb::eStateUnloaded ||
				$pid_state == $lldb::eStateAttaching ||
				$pid_state == $lldb::eStateLaunching )
			{
				
			}
		    elsif (	$pid_state == $lldb::eStateStopped )
			{
				my $tid = lldb::PDProcessGetSelectedThread ( $pid );
				my $pc = lldb::PDThreadGetRegisterHexValueByName($pid, $tid, $lldb::PD_REGISTER_SET_ALL, "eip", 0);
				$pc != 0 and printf("pc = 0x%8.8x ", $pc); 
				# my $sp = lldb::PDThreadGetRegisterHexValueByName($pid, $tid, $lldb::PD_REGISTER_SET_ALL, "esp", 0);
				# $sp != 0 and printf("sp = 0x%8.8x ", $sp);
				# my $fp = lldb::PDThreadGetRegisterHexValueByName($pid, $tid, $lldb::PD_REGISTER_SET_ALL, "ebp", 0);
				# $sp != 0 and printf("fp = 0x%8.8x ", $fp);
				# print "\n";
				my $done = 0;
				my $input;
				while (!$done)
				{
					print '(pdbg) '; 
					
					chomp($input = <STDIN>);
					my @argv = split(/\s+/, $input);
					my $cmd = @argv ? shift @argv : undef;
					my $cmd_href = get_command_hash_ref ($cmd);
					if ($cmd_href)
					{
						# Print the expanded alias if one was used
						if ($opt_v and $cmd_href->{name} ne $cmd)
						{
							print "$cmd_href->{name} @argv\n";
						}

						# Call the command's callback function to make things happen
						if ($cmd_href->{function}($pid, $tid, @argv))
						{							
							$done = $cmd_href->{runs_target};
						}
					}
					else
					{
						print "invalid command: '$cmd'\nType 'help' for a list of all commands.\nType 'help CMD' for help on a specific commmand.\n";
					}
				}
			}
		    elsif (	$pid_state == $lldb::eStateRunning ||
			 		$pid_state == $lldb::eStateStepping )
			{
				
			}
		    elsif (	$pid_state == $lldb::eStateCrashed ||
		    		$pid_state == $lldb::eStateDetached	||
		    		$pid_state == $lldb::eStateExited )
			{
				$pid = 0;				
			}
		    elsif ( $pid_state == $lldb::eStateSuspended )
			{
			}
			else
			{
			}
		}
		
		if ($pid)
		{
			$opt_g and printf("PDProcessResetEvents(%d, 0x%8.8x)\n", $pid, $events);
			lldb::PDProcessResetEvents($pid, $events);			
		}
	}	
}

if ($pid != $lldb::PD_INVALID_PROCESS_ID)
{
	lldb::PDProcessDetach ($pid);
}
