select * 
from administrative_boundaries.nuts_2021_10m nm 
where nuts_id like %(filter)s
order by levl_code 