import cgi
import cgitb

form = cgi.FieldStorage()
first = form.getvalue('first_name')
last = form.getvalue('last_name')

print("Content-type:text/html\r\n\r\n")
print("<html>")
print("<head>")
print("<title>Hello - Second CGI Program</title>")
print("</head>")
print("<body>")
print("<h2>Hello %s %s</h2>" % (first, last))
print("</body>")
print("</html>")

