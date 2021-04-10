<?php
$deger=$_GET["type"];
$type="-t ";
$command='python3 pytorch.py '.$type.$deger;
$output = shell_exec($command);
echo "<pre>$output</pre>";
?>