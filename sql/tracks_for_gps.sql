SELECT ati.track_id, ati.track_id_new, am.vehicle_id_anon as vehicle_id 
from spirite.anon_track_id ati 
join spirite.anon_master am on ati.track_id_new = am.track_id_new