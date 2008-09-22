import BaseHTTPServer
import SimpleHTTPServer
import os
import sys
import urllib, urlparse
import posixpath
import StringIO
import re
import shutil
import threading
import time
import socket

import Reporter
import ConfigParser

# Keys replaced by server.

kReportColRE = re.compile('<!-- REPORTBUGCOL -->')
kReportColRepl = '<td></td>'
kReportBugRE = re.compile('<!-- REPORTBUG id="report-(.*)\\.html" -->')         
kReportBugRepl = '<td class="View"><a href="report/\\1">Report Bug</a></td>'
kBugKeyValueRE = re.compile('<!-- BUG([^ ]*) (.*) -->')

kReportReplacements = [(kReportColRE, kReportColRepl),
                       (kReportBugRE, kReportBugRepl)]

# Other simple parameters

kResources = posixpath.join(posixpath.dirname(__file__), 'Resources')
kConfigPath = os.path.expanduser('~/.scanview.cfg')

###

__version__ = "0.1"

__all__ = ["create_server"]

class ReporterThread(threading.Thread):
    def __init__(self, report, reporter, parameters, server):
        threading.Thread.__init__(self)
        self.report = report
        self.server = server
        self.reporter = reporter
        self.parameters = parameters
        self.success = False
        self.status = None

    def run(self):
        result = None
        try:
            if self.server.options.debug:
                print >>sys.stderr, "%s: SERVER: submitting bug."%(sys.argv[0],)
            self.status = self.reporter.fileReport(self.report, self.parameters)
            self.success = True
            time.sleep(3)
            if self.server.options.debug:
                print >>sys.stderr, "%s: SERVER: submission complete."%(sys.argv[0],)
        except Reporter.ReportFailure,e:
            self.status = e.value
        except Exception,e:
            s = StringIO.StringIO()
            import traceback
            print >>s,'<b>Unhandled Exception</b><br><pre>'
            traceback.print_exc(e,file=s)
            print >>s,'</pre>'
            self.status = s.getvalue()

class ScanViewServer(BaseHTTPServer.HTTPServer):
    def __init__(self, address, handler, root, reporters, options):
        BaseHTTPServer.HTTPServer.__init__(self, address, handler)
        self.root = root
        self.reporters = reporters
        self.options = options        
        self.halted = False
        self.config = None
        self.load_config()

    def load_config(self):
        self.config = ConfigParser.RawConfigParser()

        # Add defaults
        self.config.add_section('ScanView')
        for r in self.reporters:
            self.config.add_section(r.getName())
            for p in r.getParameterNames():
                self.config.set(r.getName(), p, '')

        # Ignore parse errors
        try:
            self.config.read([kConfigPath])
        except:
            pass

        # Save on exit
        import atexit
        atexit.register(lambda: self.save_config())
        
    def save_config(self):
        # Ignore errors (only called on exit).
        try:
            f = open(kConfigPath,'w')
            self.config.write(f)
            f.close()
        except:
            pass
        
    def halt(self):
        self.halted = True
        if self.options.debug:
            print >>sys.stderr, "%s: SERVER: halting." % (sys.argv[0],)

    def serve_forever(self):
        while not self.halted:
            if self.options.debug > 1:
                print >>sys.stderr, "%s: SERVER: waiting..." % (sys.argv[0],)
            try:
                self.handle_request()
            except OSError,e:
                print 'OSError',e.errno

    def finish_request(self, request, client_address):
        if self.options.autoReload:
            import ScanView
            self.RequestHandlerClass = reload(ScanView).ScanViewRequestHandler
        BaseHTTPServer.HTTPServer.finish_request(self, request, client_address)

    def handle_error(self, request, client_address):
        # Ignore socket errors
        info = sys.exc_info()
        if info and isinstance(info[1], socket.error):
            if self.options.debug > 1:
                print >>sys.stderr, "%s: SERVER: ignored socket error." % (sys.argv[0],)
            return
        BaseHTTPServer.HTTPServer.handle_error(self, request, client_address)

# Borrowed from Quixote, with simplifications.
def parse_query(qs, fields=None):
    if fields is None:
        fields = {}
    for chunk in filter(None, qs.split('&')):
        if '=' not in chunk:
            name = chunk
            value = ''
        else:
            name, value = chunk.split('=', 1)
        name = urllib.unquote(name.replace('+', ' '))
        value = urllib.unquote(value.replace('+', ' '))
        fields[name] = value
    return fields

class ScanViewRequestHandler(SimpleHTTPServer.SimpleHTTPRequestHandler):
    server_version = "ScanViewServer/" + __version__
    dynamic_mtime = time.time()

    def do_HEAD(self):
        try:
            SimpleHTTPServer.SimpleHTTPRequestHandler.do_HEAD(self)
        except Exception,e:
            self.handle_exception(e)
            
    def do_GET(self):
        try:
            SimpleHTTPServer.SimpleHTTPRequestHandler.do_GET(self)
        except Exception,e:
            self.handle_exception(e)
            
    def do_POST(self):
        """Serve a POST request."""
        try:
            length = self.headers.getheader('content-length') or "0"
            try:
                length = int(length)
            except:
                length = 0
            content = self.rfile.read(length)
            fields = parse_query(content)
            f = self.send_head(fields)
            if f:
                self.copyfile(f, self.wfile)
                f.close()
        except Exception,e:
            self.handle_exception(e)            

    def log_message(self, format, *args):
        if self.server.options.debug:
            sys.stderr.write("%s: SERVER: %s - - [%s] %s\n" %
                             (sys.argv[0],
                              self.address_string(),
                              self.log_date_time_string(),
                              format%args))

    def load_report(self, report):
        path = os.path.join(self.server.root, 'report-%s.html'%report)
        data = open(path).read()
        keys = {}
        for item in kBugKeyValueRE.finditer(data):
            k,v = item.groups()
            keys[k] = v
        return keys

    def handle_exception(self, exc):
        import traceback
        s = StringIO.StringIO()
        print >>s, "INTERNAL ERROR\n"
        traceback.print_exc(exc, s)
        f = self.send_string(s.getvalue(), 'text/plain')
        if f:
            self.copyfile(f, self.wfile)
            f.close()        
    
    def submit_bug(self):
        title = self.fields.get('title')
        description = self.fields.get('description')
        report = self.fields.get('report')
        reporterIndex = self.fields.get('reporter')

        # Type check form parameters.
        reportPath = posixpath.join(self.server.root,
                                   'report-%s.html' % report)
        if not posixpath.exists(reportPath):
            return (False, "Invalid report ID.")
        if not title:
            return (False, "Missing title.")
        if not description:
            return (False, "Missing description.")
        try:
            reporterIndex = int(reporterIndex)
        except:
            return (False, "Invalid report method.")
        
        # Get the reporter and parameters.
        reporter = self.server.reporters[reporterIndex]
        parameters = {}
        for o in reporter.getParameterNames():
            name = '%s_%s'%(reporter.getName(),o)
            if name not in self.fields:
                return (False, 
                        'Missing field "%s" for %s report method.'%(name,
                                                                    reporter.getName()))
            parameters[o] = self.fields[name]

        # Update config defaults.
        self.server.config.set('ScanView', 'reporter', reporterIndex)
        for o in reporter.getParameterNames():
            self.server.config.set(reporter.getName(), o, parameters[o])

        # Create the report.
        bug = Reporter.BugReport(title, description, [reportPath])

        # Kick off a reporting thread.
        t = ReporterThread(bug, reporter, parameters, self.server)
        t.start()

        # Wait for thread to die...
        while t.isAlive():
            time.sleep(.25)
        submitStatus = t.status

        return (t.success, t.status)

    def send_report_submit(self):
        report = self.fields.get('report')
        title = self.fields.get('title')
        description = self.fields.get('description')

        res,message = self.submit_bug()

        if res:
            statusClass = 'SubmitOk'
            statusName = 'Succeeded'
        else:
            statusClass = 'SubmitFail'
            statusName = 'Failed'

        result = """
<head>
  <title>Bug Submission</title>
  <link rel="stylesheet" type="text/css" href="/scanview.css" />
</head>
<body>
<h3>
<a href="/">Summary</a> > 
<a href="/report-%(report)s.html">Report %(report)s</a> > 
<a href="/report/%(report)s">File Bug</a> > 
Submit</h3>
<form name="form" action="">
<table class="form">
<tr><td>
<table class="form_group">
<tr>
  <td class="form_clabel">Title:</td>
  <td class="form_value">
    <input type="text" name="title" size="50" value="%(title)s" disabled>
  </td>
</tr>
<tr>
  <td class="form_label">Description:</td>
  <td class="form_value">
<textarea rows="10" cols="80" name="description" disabled>
%(description)s
</textarea>
  </td>
</table>
</td></tr>
</table>
</form>
<h1 class="%(statusClass)s">Submission %(statusName)s</h1>
%(message)s
<p>
<hr>
<a href="/">Return to Summary</a>
</body>
</html>"""%locals()
        return self.send_string(result)

    def send_report(self, report):
        try:
            keys = self.load_report(report)
        except IOError:
            return self.send_error(400, 'Invalid report.')

        initialTitle = keys.get('DESC','')
        initialDescription = """\
Bug generated by the clang static analyzer.

Description: %s
File: %s
Line: %s
"""%(initialTitle,
     keys.get('FILE','<unknown>'),
     keys.get('LINE','<unknown>'))

        reporterSelections = []
        reporterOptions = []
        
        try:
            active = self.server.config.getint('ScanView','reporter')
        except:
            active = 0
        for i,r in enumerate(self.server.reporters):
            selected = (i == active)
            if selected:
                selectedStr = ' selected'
            else:
                selectedStr = ''
            reporterSelections.append('<option value="%d"%s>%s</option>'%(i,selectedStr,r.getName()))
            options = '\n'.join(["""\
<tr>
  <td class="form_clabel">%s:</td>
  <td class="form_value"><input type="text" name="%s_%s" value="%s"></td>
</tr>"""%(o,r.getName(),o,self.server.config.get(r.getName(), o)) for o in r.getParameterNames()])
            display = ('none','')[selected]
            reporterOptions.append("""\
<tr id="%sReporterOptions" style="display:%s">
  <td class="form_label">%s Options</td>
  <td class="form_value">
    <table class="form_inner_group">
%s
    </table>
  </td>
</tr>
"""%(r.getName(),display,r.getName(),options))
        reporterSelections = '\n'.join(reporterSelections)
        reporterOptionsDivs = '\n'.join(reporterOptions)
        reportersArray = '[%s]'%(','.join([`r.getName()` for r in self.server.reporters]))

        result = """<html>
<head>
  <title>File Bug</title>
  <link rel="stylesheet" type="text/css" href="/scanview.css" />
</head>
<script language="javascript" type="text/javascript">
var reporters = %(reportersArray)s;
function updateReporterOptions() {
  index = document.getElementById('reporter').selectedIndex;
  for (var i=0; i < reporters.length; ++i) {
    o = document.getElementById(reporters[i] + "ReporterOptions");
    if (i == index) {
      o.style.display = "";
    } else {
      o.style.display = "none";
    }
  }
}
</script>
<body onLoad="updateReporterOptions()">
<h3>
<a href="/">Summary</a> > 
<a href="/report-%(report)s.html">Report %(report)s</a> > 
File Bug</h3>
<form name="form" action="/report_submit" method="post">
<input type="hidden" name="report" value="%(report)s">

<table class="form">
<tr><td>
<table class="form_group">
<tr>
  <td class="form_clabel">Title:</td>
  <td class="form_value">
    <input type="text" name="title" size="50" value="%(initialTitle)s">
  </td>
</tr>
<tr>
  <td class="form_label">Description:</td>
  <td class="form_value">
<textarea rows="10" cols="80" name="description">
%(initialDescription)s
</textarea>
  </td>
</table>
<br>
<table class="form_group">
<tr>
  <td class="form_clabel">Method:</td>
  <td class="form_value">
    <select id="reporter" name="reporter" onChange="updateReporterOptions()">
    %(reporterSelections)s
    </select>
  </td>
</tr>
%(reporterOptionsDivs)s
</table>
<br>
</td></tr>
<tr><td class="form_submit">
  <input align="right" type="submit" name="Submit" value="Submit">
</td></tr>
</table>
</form>

<iframe src="/report-%(report)s.html" width="100%%" height="40%%"
        scrolling="auto" frameborder="1">
  <a href="/report-%(report)s.html">View Bug Report</a>
</iframe>

</body>
</html>"""%locals()
        return self.send_string(result)

    def send_head(self, fields=None):
        if fields is None:
            fields = {}
        self.fields = fields

        o = urlparse.urlparse(self.path)
        self.fields = parse_query(o.query, fields)
        path = posixpath.normpath(urllib.unquote(o.path))

        # Split the components and strip the root prefix.
        components = path.split('/')[1:]
        
        # Special case some top-level entries.
        if components:
            name = components[0]
            if len(components)==2:
                if name=='report':
                    return self.send_report(components[1])
            elif len(components)==1:
                if name=='quit':
                    self.server.halt()
                    return self.send_string('Goodbye.', 'text/plain')
                elif name=='report_submit':
                    return self.send_report_submit()
                elif name=='favicon.ico':
                    return self.send_path(posixpath.join(kResources,'bugcatcher.ico'))
                elif name=='scanview.css':
                    return self.send_path(posixpath.join(kResources,'scanview.css'))
        
        # Match directory entries.
        if components[-1] == '':
            components[-1] = 'index.html'

        suffix = '/'.join(components)

        # The summary may reference source files on disk using rooted
        # paths. Make sure these resolve correctly for now.
        # FIXME: This isn't a very good idea... we should probably
        # mark rooted paths somehow.        
        if os.path.exists(posixpath.join('/', suffix)):
            path = posixpath.join('/', suffix)
        else:
            path = posixpath.join(self.server.root, suffix)

        if self.server.options.debug > 1:
            print >>sys.stderr, '%s: SERVER: sending path "%s"'%(sys.argv[0],
                                                                 path)
        return self.send_path(path)

    def send_404(self):
        self.send_error(404, "File not found")
        return None

    def send_path(self, path):
        ctype = self.guess_type(path)
        if ctype.startswith('text/'):
            # Patch file instead
            return self.send_patched_file(path, ctype)
        else:
            mode = 'rb'
        try:
            f = open(path, mode)
        except IOError:
            return self.send_404()
        return self.send_file(f, ctype)

    def send_file(self, f, ctype):
        # Patch files to add links, but skip binary files.
        self.send_response(200)
        self.send_header("Content-type", ctype)
        fs = os.fstat(f.fileno())
        self.send_header("Content-Length", str(fs[6]))
        self.send_header("Last-Modified", self.date_time_string(fs.st_mtime))
        self.end_headers()
        return f

    def send_string(self, s, ctype='text/html', headers=True, mtime=None):
        if headers:
            self.send_response(200)
            self.send_header("Content-type", ctype)
            self.send_header("Content-Length", str(len(s)))
            if mtime is None:
                mtime = self.dynamic_mtime
            self.send_header("Last-Modified", self.date_time_string(mtime))
            self.end_headers()
        return StringIO.StringIO(s)

    def send_patched_file(self, path, ctype):
        try:
            f = open(path,'r')
        except IOError:
            return self.send_404()
        fs = os.fstat(f.fileno())
        data = f.read()
        for a,b in kReportReplacements:
            data = a.sub(b, data)
        return self.send_string(data, ctype, mtime=fs.st_mtime)


def create_server(address, options, root):
    import Reporter

    reporters = Reporter.getReporters()

    return ScanViewServer(address, ScanViewRequestHandler,
                          root,
                          reporters,
                          options)
