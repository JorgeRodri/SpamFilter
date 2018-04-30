body = '''<html>
<head>
<style>
table, th, td {{
    border: 1px solid black;
    border-collapse: collapse;
}}
th, td {{
    padding: 5px;
    text-align: left;
}}
</style>
</head>
<body>

<table style="width:50%">
  <tr>
    <th>     </th>
    <th>PredecidoNoSpam</th>
    <th>PredecidoSpam</th>
  </tr>
  <tr>
  	<th>No spam</th>
    <td>{0}</td>
    <td>{1}</td>
  </tr>
  <tr>
  	<th>Spam</th>
    <td>{2}</td>
    <td>{3}</td>
  </tr>
</table>

</body>
</html>
'''