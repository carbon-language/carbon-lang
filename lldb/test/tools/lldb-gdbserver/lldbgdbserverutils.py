"""Module for supporting unit testing of the lldb-gdbserver debug monitor exe.
"""

import os
import os.path
import platform
import re
import select
import subprocess
import time


def _get_debug_monitor_from_lldb(lldb_exe, debug_monitor_basename):
    """Return the debug monitor exe path given the lldb exe path.

    This method attempts to construct a valid debug monitor exe name
    from a given lldb exe name.  It will return None if the synthesized
    debug monitor name is not found to exist.

    The debug monitor exe path is synthesized by taking the directory
    of the lldb exe, and replacing the portion of the base name that
    matches "lldb" (case insensitive) and replacing with the value of
    debug_monitor_basename.

    Args:
        lldb_exe: the path to an lldb executable.

        debug_monitor_basename: the base name portion of the debug monitor
            that will replace 'lldb'.

    Returns:
        A path to the debug monitor exe if it is found to exist; otherwise,
        returns None.

    """

    exe_dir = os.path.dirname(lldb_exe)
    exe_base = os.path.basename(lldb_exe)

    # we'll rebuild the filename by replacing lldb with
    # the debug monitor basename, keeping any prefix or suffix in place.
    regex = re.compile(r"lldb", re.IGNORECASE)
    new_base = regex.sub(debug_monitor_basename, exe_base)

    debug_monitor_exe = os.path.join(exe_dir, new_base)
    if os.path.exists(debug_monitor_exe):
        return debug_monitor_exe
    else:
        return None


def get_lldb_gdbserver_exe():
    """Return the lldb-gdbserver exe path.

    Returns:
        A path to the lldb-gdbserver exe if it is found to exist; otherwise,
        returns None.
    """
    lldb_exe = os.environ["LLDB_EXEC"]
    if not lldb_exe:
        return None
    else:
        return _get_debug_monitor_from_lldb(lldb_exe, "lldb-gdbserver")

def get_debugserver_exe():
    """Return the debugserver exe path.

    Returns:
        A path to the debugserver exe if it is found to exist; otherwise,
        returns None.
    """
    lldb_exe = os.environ["LLDB_EXEC"]
    if not lldb_exe:
        return None
    else:
        return _get_debug_monitor_from_lldb(lldb_exe, "debugserver")


_LOG_LINE_REGEX = re.compile(r'^(lldb-gdbserver|debugserver)\s+<\s*(\d+)>' +
    '\s+(read|send)\s+packet:\s+(.+)$')


def _is_packet_lldb_gdbserver_input(packet_type, llgs_input_is_read):
    """Return whether a given packet is input for lldb-gdbserver.

    Args:
        packet_type: a string indicating 'send' or 'receive', from a
            gdbremote packet protocol log.

        llgs_input_is_read: true if lldb-gdbserver input (content sent to
            lldb-gdbserver) is listed as 'read' or 'send' in the packet
            log entry.

    Returns:
        True if the packet should be considered input for lldb-gdbserver; False
        otherwise.
    """
    if packet_type == 'read':
        # when llgs is the read side, then a read packet is meant for
        # input to llgs (when captured from the llgs/debugserver exe).
        return llgs_input_is_read
    elif packet_type == 'send':
        # when llgs is the send side, then a send packet is meant to
        # be input to llgs (when captured from the lldb exe).
        return not llgs_input_is_read
    else:
        # don't understand what type of packet this is
        raise "Unknown packet type: {}".format(packet_type)


_STRIP_CHECKSUM_REGEX = re.compile(r'#[0-9a-fA-F]{2}$')

def assert_packets_equal(asserter, actual_packet, expected_packet):
    # strip off the checksum digits of the packet.  When we're in
    # no-ack mode, the # checksum is ignored, and should not be cause
    # for a mismatched packet.
    actual_stripped = _STRIP_CHECKSUM_REGEX.sub('', actual_packet)
    expected_stripped = _STRIP_CHECKSUM_REGEX.sub('', expected_packet)
    asserter.assertEqual(actual_stripped, expected_stripped)


_GDB_REMOTE_PACKET_REGEX = re.compile(r'^\$([^\#]*)#[0-9a-fA-F]{2}')

def expect_lldb_gdbserver_replay(
    asserter,
    sock,
    test_sequence,
    timeout_seconds,
    logger=None):
    """Replay socket communication with lldb-gdbserver and verify responses.

    Args:
        asserter: the object providing assertEqual(first, second, msg=None), e.g. TestCase instance.

        sock: the TCP socket connected to the lldb-gdbserver exe.

        test_sequence: a GdbRemoteTestSequence instance that describes
            the messages sent to the gdb remote and the responses
            expected from it.

        timeout_seconds: any response taking more than this number of
           seconds will cause an exception to be raised.

        logger: a Python logger instance.

    Returns:
        The context dictionary from running the given gdbremote
        protocol sequence.  This will contain any of the capture
        elements specified to any GdbRemoteEntry instances in
        test_sequence.
    """
    
    # Ensure we have some work to do.
    if len(sequence_entry) < 1:
        return {}
    
    received_lines = []
    receive_buffer = ''
    context = {}

    sequence_entry = test_sequence.entries.pop(0)
    while sequence_entry:
        if sequence_entry.is_send_to_remote():
            # This is an entry to send to the remote debug monitor.
            if logger:
                logger.info("sending packet to remote: {}".format(sequence_entry.exact_payload))
            sock.sendall(sequence_entry.get_send_packet())
        else:
            # This is an entry to expect to receive from the remote debug monitor.
            if logger:
                logger.info("receiving packet from remote, should match: {}".format(sequence_entry.exact_payload))

            start_time = time.time()
            timeout_time = start_time + timeout_seconds

            # while we don't have a complete line of input, wait
            # for it from socket.
            while len(received_lines) < 1:
                # check for timeout
                if time.time() > timeout_time:
                    raise Exception(
                        'timed out after {} seconds while waiting for llgs to respond with: {}, currently received: {}'.format(
                            timeout_seconds, sequence_entry.exact_payload, receive_buffer))
                can_read, _, _ = select.select([sock], [], [], 0)
                if can_read and sock in can_read:
                    new_bytes = sock.recv(4096)
                    if new_bytes and len(new_bytes) > 0:
                        # read the next bits from the socket
                        if logger:
                            logger.debug("llgs responded with bytes: {}".format(new_bytes))
                        receive_buffer += new_bytes

                        # parse fully-formed packets into individual packets
                        has_more = len(receive_buffer) > 0
                        while has_more:
                            if len(receive_buffer) <= 0:
                                has_more = False
                            # handle '+' ack
                            elif receive_buffer[0] == '+':
                                received_lines.append('+')
                                receive_buffer = receive_buffer[1:]
                                if logger:
                                    logger.debug('parsed packet from llgs: +, new receive_buffer: {}'.format(receive_buffer))
                            else:
                                packet_match = _GDB_REMOTE_PACKET_REGEX.match(receive_buffer)
                                if packet_match:
                                    received_lines.append(packet_match.group(0))
                                    receive_buffer = receive_buffer[len(packet_match.group(0)):]
                                    if logger:
                                        logger.debug('parsed packet from llgs: {}, new receive_buffer: {}'.format(packet_match.group(0), receive_buffer))
                                else:
                                    has_more = False
            # got a line - now try to match it against expected line
            if len(received_lines) > 0:
                received_packet = received_lines.pop(0)
                context = sequence_entry.assert_match(asserter, received_packet, context=context)
                
        # Move on to next sequence entry as needed.  Some sequence entries support executing multiple
        # times in different states (for looping over query/response packets).
        if sequence_entry.is_consumed():
            if len(test_sequence.entries) > 0:
                sequence_entry = test_sequence.entries.pop(0)
            else:
                sequence_entry = None
    return context


def gdbremote_hex_encode_string(str):
    output = ''
    for c in str:
        output += '{0:02x}'.format(ord(c))
    return output


def gdbremote_packet_encode_string(str):
    checksum = 0
    for c in str:
        checksum += ord(c)
    return '$' + str + '#{0:02x}'.format(checksum % 256)


def build_gdbremote_A_packet(args_list):
    """Given a list of args, create a properly-formed $A packet containing each arg.
    """
    payload = "A"

    # build the arg content
    arg_index = 0
    for arg in args_list:
        # Comma-separate the args.
        if arg_index > 0:
            payload += ','

        # Hex-encode the arg.
        hex_arg = gdbremote_hex_encode_string(arg)

        # Build the A entry.
        payload += "{},{},{}".format(len(hex_arg), arg_index, hex_arg)

        # Next arg index, please.
        arg_index += 1

    # return the packetized payload
    return gdbremote_packet_encode_string(payload)

class GdbRemoteEntry(object):

    def __init__(self, is_send_to_remote=True, exact_payload=None, regex=None, capture=None, expect_captures=None):
        """Create an entry representing one piece of the I/O to/from a gdb remote debug monitor.

        Args:

            is_send_to_remote: True if this entry is a message to be
                sent to the gdbremote debug monitor; False if this
                entry represents text to be matched against the reply
                from the gdbremote debug monitor.

            exact_payload: if not None, then this packet is an exact
                send (when sending to the remote) or an exact match of
                the response from the gdbremote. The checksums are
                ignored on exact match requests since negotiation of
                no-ack makes the checksum content essentially
                undefined.

            regex: currently only valid for receives from gdbremote.
                When specified (and only if exact_payload is None),
                indicates the gdbremote response must match the given
                regex. Match groups in the regex can be used for two
                different purposes: saving the match (see capture
                arg), or validating that a match group matches a
                previously established value (see expect_captures). It
                is perfectly valid to have just a regex arg and to
                specify neither capture or expect_captures args. This
                arg only makes sense if exact_payload is not
                specified.

            capture: if specified, is a dictionary of regex match
                group indices (should start with 1) to variable names
                that will store the capture group indicated by the
                index. For example, {1:"thread_id"} will store capture
                group 1's content in the context dictionary where
                "thread_id" is the key and the match group value is
                the value. The value stored off can be used later in a
                expect_captures expression. This arg only makes sense
                when regex is specified.

            expect_captures: if specified, is a dictionary of regex
                match group indices (should start with 1) to variable
                names, where the match group should match the value
                existing in the context at the given variable name.
                For example, {2:"thread_id"} indicates that the second
                match group must match the value stored under the
                context's previously stored "thread_id" key. This arg
                only makes sense when regex is specified.
        """
        self.is_send_to_remote = is_send_to_remote
        self.exact_payload = exact_payload
        self.regex = regex
        self.capture = capture
        self.expect_captures = expect_captures

    def is_send_to_remote(self):
        return self.is_send_to_remote

    def _assert_exact_payload_match(self, asserter, actual_packet):
        assert_packets_equal(asserter, actual_packet, self.exact_payload)
        return None

    def _assert_regex_match(self, asserter, actual_packet, context):
        # Ensure the actual packet matches from the start of the actual packet.
        match = self.regex.match(actual_packet)
        asserter.assertIsNotNone(match)

        if self.capture:
            # Handle captures.
            for group_index, var_name in self.capture.items():
                capture_text = match.group(group_index)
                if not capture_text:
                    raise Exception("No content for group index {}".format(group_index))
                context[var_name] = capture_text

        if self.expect_captures:
            # Handle comparing matched groups to context dictionary entries.
            for group_index, var_name in self.expect_captures.items():
                capture_text = match.group(group_index)
                if not capture_text:
                    raise Exception("No content to expect for group index {}".format(group_index))
                asserter.assertEquals(capture_text, context[var_name])

        return context

    def assert_match(self, asserter, actual_packet, context=None):
        # This only makes sense for matching lines coming from the
        # remote debug monitor.
        if self.is_send_to_remote:
            raise Exception("Attempted to match a packet being sent to the remote debug monitor, doesn't make sense.")

        # Create a new context if needed.
        if not context:
            context = {}

        # If this is an exact payload, ensure they match exactly,
        # ignoring the packet checksum which is optional for no-ack
        # mode.
        if self.exact_payload:
            self._assert_exact_payload_match(asserter, actual_packet)
            return context
        elif self.regex:
            return self._assert_regex_match(asserter, actual_packet, context)
        else:
            raise Exception("Don't know how to match a remote-sent packet when exact_payload isn't specified.")

class GdbRemoteTestSequence(object):

    _LOG_LINE_REGEX = re.compile(r'^.*(read|send)\s+packet:\s+(.+)$')

    def __init__(self, logger):
        self.entries = []
        self.logger = logger

    def add_log_lines(self, log_lines, remote_input_is_read):
        for line in log_lines:
            if type(line) == str:
                # Handle log line import
                if self.logger:
                    self.logger.debug("processing log line: {}".format(line))
                match = self._LOG_LINE_REGEX.match(line)
                if match:
                    playback_packet = match.group(2)
                    direction = match.group(1)
                    if _is_packet_lldb_gdbserver_input(direction, remote_input_is_read):
                        # Handle as something to send to the remote debug monitor.
                        if self.logger:
                            self.logger.info("processed packet to send to remote: {}".format(playback_packet))
                        self.entries.append(GdbRemoteEntry(is_send_to_remote=True, exact_payload=playback_packet))
                    else:
                        # Log line represents content to be expected from the remote debug monitor.
                        if self.logger:
                            self.logger.info("receiving packet from llgs, should match: {}".format(playback_packet))
                        self.entries.append(GdbRemoteEntry(is_send_to_remote=False,exact_payload=playback_packet))
                else:
                    raise Exception("failed to interpret log line: {}".format(line))
            elif type(line) == dict:
                # Handle more explicit control over details via dictionary.
                direction = line.get("direction", None)
                regex = line.get("regex", None)
                capture = line.get("capture", None)
                expect_captures = line.get("expect_captures", None)

                # Compile the regex.
                if regex and (type(regex) == str):
                    regex = re.compile(regex)

                if _is_packet_lldb_gdbserver_input(direction, remote_input_is_read):
                    # Handle as something to send to the remote debug monitor.
                    if self.logger:
                        self.logger.info("processed dict sequence to send to remote")
                    self.entries.append(GdbRemoteEntry(is_send_to_remote=True, regex=regex, capture=capture, expect_captures=expect_captures))
                else:
                    # Log line represents content to be expected from the remote debug monitor.
                    if self.logger:
                        self.logger.info("processed dict sequence to match receiving from remote")
                    self.entries.append(GdbRemoteEntry(is_send_to_remote=False, regex=regex, capture=capture, expect_captures=expect_captures))

def process_is_running(pid, unknown_value=True):
    """If possible, validate that the given pid represents a running process on the local system.

    Args:

        pid: an OS-specific representation of a process id.  Should be an integral value.

        unknown_value: value used when we cannot determine how to check running local
        processes on the OS.

    Returns:

        If we can figure out how to check running process ids on the given OS:
        return True if the process is running, or False otherwise.

        If we don't know how to check running process ids on the given OS:
        return the value provided by the unknown_value arg.
    """
    if type(pid) != int:
        raise Exception("pid must be of type int")

    process_ids = []

    if platform.system() in ['Darwin', 'Linux', 'FreeBSD', 'NetBSD']:
        # Build the list of running process ids
        output = subprocess.check_output("ps ax | awk '{ print $1; }'", shell=True)
        text_process_ids = output.split('\n')[1:]
        # Convert text pids to ints
        process_ids = [int(text_pid) for text_pid in text_process_ids if text_pid != '']
    # elif {your_platform_here}:
    #   fill in process_ids as a list of int type process IDs running on
    #   the local system.
    else:
        # Don't know how to get list of running process IDs on this
        # OS, so return the "don't know" value.
        return unknown_value

    # Check if the pid is in the process_ids
    return pid in process_ids

if __name__ == '__main__':
    EXE_PATH = get_lldb_gdbserver_exe()
    if EXE_PATH:
        print "lldb-gdbserver path detected: {}".format(EXE_PATH)
    else:
        print "lldb-gdbserver could not be found"
