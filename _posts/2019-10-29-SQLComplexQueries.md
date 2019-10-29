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

## What is the query which reports the consistency between orders and their attached invoices?  
The resultset should report for each (CustomerID, CustomerName)
 1. the total number of orders: TotalNBOrders
 2. the number of invoices converted from an order: TotalNBInvoices
 3. the total value of orders: OrdersTotalValue
 4. the total value of invoices: InvoicesTotalValue
 5. the absolute value of the difference between c - d: AbsoluteValueDifference   
   
 Here is a screenshot of the expected resultset:  
![resultset](/images/SoftwareEngineering/SQLComplexQuery01.jpg "Query 1 resultset.")

The tables involved in the query and their links are shown in the screenshot below:  
![DB Schema subset](/images/SoftwareEngineering/DB-Schema01.png "Query 1 schema.")

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

## Update a specific UnitPrice of a product   
For the CustomerId = 1060 (CustomerName = 'Anand Mudaliyar')   
Identify the first InvoiceLine of his first Invoice, where "first" means the lowest respective IDs, and write an update query increasing the UnitPrice of this InvoiceLine by 20.

A screenshot of the expected resultset after the update query is shown below:    
![resultset02](/images/SoftwareEngineering/SQLComplexQuery020.jpg "Query 2 resultset.")   

The SQL query is listed below:   

```sql
UPDATE Sales.InvoiceLines
SET Sales.InvoiceLines.UnitPrice = Sales.InvoiceLines.UnitPrice + 20
WHERE
Sales.InvoiceLines.InvoiceLineID = 
(
	SELECT MIN(MIL.InvoiceLineID)
	FROM Sales.InvoiceLines AS MIL
	WHERE MIL.InvoiceID =
	(
		SELECT MIN(I.InvoiceId)  
		FROM Sales.Invoices AS I, Sales.InvoiceLines AS IL
		WHERE 
			I.InvoiceID = IL.InvoiceID
			AND I.CustomerID = 1060
	)
)
```

## Create a T-SQL stored procedure to report customer's turnover 
Here are the specifications of this stored procedure: 
Using the database WideWorldImporters, write a T-SQL stored procedure called ReportCustomerTurnover.  
This procedure takes two parameters: Choice and Year, both integers.  

When Choice = 1 and Year = <aYear>, ReportCustomerTurnover selects all the customer names and their total monthly turnover (invoiced value) for the year <aYear>.  

When Choice = 2 and Year = <aYear>, ReportCustomerTurnover  selects all the customer names and their total quarterly (3 months) turnover (invoiced value) for the year <aYear>.  

When Choice = 3, the value of Year is ignored and ReportCustomerTurnover  selects all the customer names and their total yearly turnover (invoiced value).   

When no value is provided for the parameter Choice, the default value of Choice must be 1.   
When no value is provided for the parameter Year, the default value is 2013. This doesn't impact Choice = 3.  

For Choice = 3, the years can be hard-coded within the range of [2013-2016].   

NULL values in the resultsets are not acceptable and must be substituted to 0.   

All output resultsets are ordered by customer names alphabetically.      
   
    
Here's the code of the stored procedure:  
```sql
USE [WideWorldImporters]
GO
/****** Object:  StoredProcedure [dbo].[ReportCustomerTurnover]    Script Date: 10/29/2019 8:11:50 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
-- =============================================
-- Author:		<Author,,Name>
-- Create date: <Create Date,,>
-- Description:	<Description,,>
-- =============================================
ALTER PROCEDURE [dbo].[ReportCustomerTurnover] 
	-- Add the parameters for the stored procedure here
	@Choice  int=1, 
	@Year int = 2013
AS
BEGIN
	-- SET NOCOUNT ON added to prevent extra result sets from
	-- interfering with SELECT statements.
	SET NOCOUNT ON;

    -- Insert statements for procedure here
	DECLARE @TurnOver VARCHAR;
	SET @TurnOver = 'ToTalTurnOver' + CAST(@YEAR AS VARCHAR) ;
	
	IF @Choice = 1 AND  NOT(@Year IS NULL)
	
	BEGIN
		SELECT DISTINCT C.CustomerName, 

			   SUM(CASE  WHEN MONTH(T.InvoiceDate)= 1 THEN (T.UnitPrice * T.Quantity) ELSE 0 END) As Jan ,
			   SUM(CASE  WHEN MONTH(T.InvoiceDate)= 2 THEN (T.UnitPrice * T.Quantity) ELSE 0 END) AS Feb,
			   SUM(CASE  WHEN MONTH(T.InvoiceDate)= 3 THEN (T.UnitPrice * T.Quantity) ELSE 0 END) AS Mar,
			   SUM(CASE  WHEN MONTH(T.InvoiceDate)= 4 THEN (T.UnitPrice * T.Quantity) ELSE 0 END) AS Apr,
			   SUM(CASE  WHEN MONTH(T.InvoiceDate)= 5 THEN (T.UnitPrice * T.Quantity) ELSE 0 END) AS May,
			   SUM(CASE  WHEN MONTH(T.InvoiceDate)= 6 THEN (T.UnitPrice * T.Quantity) ELSE 0 END) AS Jun,
			   SUM(CASE  WHEN MONTH(T.InvoiceDate)= 7 THEN (T.UnitPrice * T.Quantity) ELSE 0 END) AS Jul,
			   SUM(CASE  WHEN MONTH(T.InvoiceDate)= 8 THEN (T.UnitPrice * T.Quantity) ELSE 0 END) AS Aug,
			   SUM(CASE  WHEN MONTH(T.InvoiceDate)= 9 THEN (T.UnitPrice * T.Quantity) ELSE 0 END) AS Sep,
			   SUM(CASE  WHEN MONTH(T.InvoiceDate)= 10 THEN (T.UnitPrice * T.Quantity) ELSE 0 END) AS Oct,
			   SUM(CASE  WHEN MONTH(T.InvoiceDate)= 11 THEN (T.UnitPrice * T.Quantity) ELSE 0 END) As Nov,
			   SUM(CASE  WHEN MONTH(T.InvoiceDate)= 12 THEN (T.UnitPrice * T.Quantity) ELSE 0 END) AS [Dec]

		FROM
		(	SELECT I.InvoiceID, I.CustomerID, I.InvoiceDate, IL.UnitPrice, IL.Quantity, IL.InvoiceLineID

   
			FROM Sales.Invoices AS I, Sales.InvoiceLines AS IL
			WHERE I.InvoiceID = IL.InvoiceID
		) AS T, Sales.Customers AS C 
		WHERE T.CustomerID = C.CustomerID
		AND YEAR(T.InvoiceDate) = @Year
		GROUP BY CustomerName  
		ORDER BY CustomerName ;
	END;
	IF @Choice = 2 AND  NOT(@Year IS NULL)
	BEGIN
		SELECT C.CustomerName, 

			   SUM(CASE  WHEN DATEPART(qq,T.InvoiceDate) = 1 THEN (T.UnitPrice * T.Quantity) ELSE 0 END) As Q1 ,
			   SUM(CASE  WHEN DATEPART(qq,T.InvoiceDate) = 2 THEN (T.UnitPrice * T.Quantity) ELSE 0 END) AS Q2 ,
			   SUM(CASE  WHEN DATEPART(qq,T.InvoiceDate) = 3 THEN (T.UnitPrice * T.Quantity) ELSE 0 END) AS Q3,
			   SUM(CASE  WHEN DATEPART(qq,T.InvoiceDate) = 4 THEN (T.UnitPrice * T.Quantity) ELSE 0 END) AS Q4

		FROM
		(	SELECT I.InvoiceID, I.CustomerID, I.InvoiceDate, IL.UnitPrice, IL.Quantity, IL.InvoiceLineID

   
			FROM Sales.Invoices AS I, Sales.InvoiceLines AS IL
			WHERE I.InvoiceID = IL.InvoiceID
		) AS T, Sales.Customers AS C 
		WHERE T.CustomerID = C.CustomerID
		AND YEAR(T.InvoiceDate) = @Year
		GROUP BY CustomerName 	
		ORDER BY CustomerName ;
	END;
	IF @Choice = 3
	BEGIN
		SELECT C.CustomerName, 

			   SUM(CASE  WHEN YEAR(T.InvoiceDate) = 2013 THEN (T.UnitPrice * T.Quantity) ELSE 0 END) AS '2013' ,
			   SUM(CASE  WHEN YEAR(T.InvoiceDate) = 2014 THEN (T.UnitPrice * T.Quantity) ELSE 0 END) AS '2014' ,
			   SUM(CASE  WHEN YEAR(T.InvoiceDate) = 2015 THEN (T.UnitPrice * T.Quantity) ELSE 0 END) AS '2015',
			   SUM(CASE  WHEN YEAR(T.InvoiceDate) = 2016 THEN (T.UnitPrice * T.Quantity) ELSE 0 END) AS '2016'

		FROM
		(	SELECT I.InvoiceID, I.CustomerID, I.InvoiceDate, IL.UnitPrice, IL.Quantity, IL.InvoiceLineID

   
			FROM Sales.Invoices AS I, Sales.InvoiceLines AS IL
			WHERE I.InvoiceID = IL.InvoiceID
		) AS T, Sales.Customers AS C 
		WHERE T.CustomerID = C.CustomerID
		GROUP BY CustomerName 	
		ORDER BY CustomerName ;
	END;

END
```   
## How to write a SQL query which reports the highest loss of money from orders not being converted into invoices?
 In the database WideWorldImporters, write a SQL query which reports the highest loss of money from orders not being converted into invoices, by customer category. The name and id of the customer who generated this highest loss must also be identified. The resultset is ordered by highest loss.  
 
 A screenshot of the expected resultset after the update query is shown below:    
![resultset02](/images/SoftwareEngineering/SQLComplexQuery04.jpg "Query 2 resultset.")   

The SQL query is listed below:   
```sql 
SELECT  D.CustomerCategoryName, D.MaxLoss, D.CustomerName, D.CustomerID
FROM
(
	SELECT DISTINCT S.CustomerCategoryName, S.MaxLoss, S.CustomerName, S.CustomerID, ROW_NUMBER() OVER (Partition by S.CustomerCategoryName  
		            Order by S.MaxLoss DESC) AS RowNo 
	FROM
	(
		SELECT CustomerCategoryName, SUM(F.UnitPrice * F.Quantity)  OVER ( Partition by CustomerCategoryName, F.CustomerName) AS MaxLoss, 
				F.CustomerName , F.CustomerID
		FROM
		(
			SELECT  C.CustomerName, C.CustomerId, C.CustomerCategoryId, L.UnitPrice, L.Quantity
			FROM
			(
				SELECT  T.CustomerID, T.OrderID, OL.UnitPrice, OL.Quantity
				FROM 
				(
					SELECT O.CustomerID, O.OrderID
					FROM Sales.Orders as O
					WHERE NOT EXISTS
					(
						SELECT *
						FROM Sales.Invoices as I
						WHERE I.OrderID = O.OrderID
					)
				) AS T, Sales.OrderLines AS OL
				WHERE T.OrderID = OL.OrderID
			) AS L, Sales.Customers AS C
			WHERE L.CustomerID = C.CustomerID
		) AS F, Sales.CustomerCategories AS G
		WHERE F.CustomerCategoryID = G.CustomerCategoryID
	) AS S 
) AS D
WHERE D.RowNo <=1
ORDER BY D.MaxLoss DESC
``` 









