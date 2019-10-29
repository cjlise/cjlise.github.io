---
title: "Some complex SQL queries"
date: 2019-10-29
tags: [Software Engineering, SQL, RDBMS, SQLServer, Database]
header: 
   image: "/images/DeepLearning/web-3706561_200.jpg"
excerpt: "Software Engineering, SQL, RDBMS, SQLServer, Database"
---

# Some complex SQL queries
The SQL queries below are some examples of complex SQL queries mainly on the sample Microsoft database [Wide World Importers sample database](https://github.com/Microsoft/sql-server-samples/releases/tag/wide-world-importers-v1.0) and running on SQLServer. 

* Returns a resultset  with the following information: 
|CustomerID|	CustomerName|	TotalNBOrders|	TotalNBInvoices|	OrdersTotalValue|	InvoicesTotalValue|	AbsoluteValueDifference|
----------------------------------------------------------------------------------------------------------
|1060 |	Anand Mudaliyar |	4	|4	|7240.2|	7240.2|	0.00|

|1059 |	Jibek Juniskyzy	|8	|8	|13208.1	|13208.1 |	0.00|

|1061 |Agrita Abele	    |9	|9	|22829.65	|22829.65 |	0.00|


...

The SQL query is listed below: 
```sql
SELECT A.CustomerId, C.CustomerName,  COUNT( DISTINCT A.OrderId) TotalNBOrders, COUNT( DISTINCT A.InvoiceId) TotalNBInvoices,
       SUM(A.UnitPrice*A.Quantity)AS OrdersTotalValue,  SUM(A.UnitPriceI * A.QuantityI) AS InvoicesTotalValue,
	   ABS(SUM(A.UnitPrice * A.Quantity) -  SUM(A.UnitPriceI*A.QuantityI)) AS AbsoluteValueDifference
FROM 
(
	SELECT O.CustomerID, O.OrderId, NULL AS InvoiceID, OL.UnitPrice, OL.Quantity, 0 AS UnitPriceI, 0 AS QuantityI, OL.OrderLineID, NULL AS InvoiceLineID 
	FROM Sales.Orders As O, Sales.OrderLines AS OL
	WHERE O.OrderId = OL.OrderID AND EXISTS
	(	SELECT II.OrderId
		FROM Sales.Invoices AS II
		WHERE II.OrderID = O.OrderID
	)
	UNION
	SELECT I.CustomerID, NULL AS OrderId, I.InvoiceID, 0 AS UnitPriceO, 0 AS QuantityO, IL.UnitPrice, IL.Quantity, NULL AS OrderLineID, InvoiceLineID
	FROM Sales.Invoices AS I, Sales.InvoiceLines AS IL
	WHERE I.InvoiceID = IL.InvoiceID
) AS A, Sales.Customers As C
WHERE A.CustomerID = C.CustomerID
GROUP BY A.CustomerID, C.CustomerName
ORDER BY AbsoluteValueDifference DESC, TotalNBOrders, CustomerName

```




