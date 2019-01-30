#ifndef GPS_H
#define GPS_H
#include <math.h>
#define DEG2RAD(a)	((a) / (180 / M_PI))
#define RAD2DEG(a)	((a) * (180 / M_PI))
#define EARTH_RADIUS 6378137 // meters
// US
//static double baseLat = DEG2RAD(32.694052), baseLon = DEG2RAD(-113.958389);
// Beijing
// static double baseLat = DEG2RAD(39.714178), baseLon = DEG2RAD(117.305466);
// Pudong
static double baseLat = DEG2RAD(30.9081441476), baseLon = DEG2RAD(121.947248386);
// latitude and longitude are in degrees(-180~180), not (-pi/2~pi/2)
static inline void latlon2xy(double lat, double lon, double &x, double &y)
{
// rotate east-west first and then north-south
	lat = DEG2RAD(lat), lon = DEG2RAD(lon);
	double xx = cos(lat)*cos(lon)*cos(baseLon)*cos(baseLat) + cos(lat)*sin(lon)*sin(baseLon)*cos(baseLat) + sin(lat)*sin(baseLat),
		yy = -cos(lat)*cos(lon)*sin(baseLon) + cos(lat)*sin(lon)*cos(baseLon),
		zz = -cos(lat)*cos(lon)*cos(baseLon)*sin(baseLat) - cos(lat)*sin(lon)*sin(baseLon)*sin(baseLat) + sin(lat)*cos(baseLat);
	x = atan2(yy, xx) * EARTH_RADIUS, y = log(tan(asin(zz) / 2 + M_PI/4 )) * EARTH_RADIUS;
}
static inline void xy2latlon(double x, double y, double & lat, double & lon)
{
// rotate north-south first and then east-west
	lon = x/EARTH_RADIUS, lat = 2 * atan(exp( y/EARTH_RADIUS)) - M_PI/2;
	double xx = cos(lat)*cos(lon)*cos(-baseLon)*cos(-baseLat) + cos(lat)*sin(lon)*sin(-baseLon) + sin(lat)*cos(-baseLon)*sin(-baseLat),
		yy = -cos(lat)*cos(lon)*cos(-baseLat)*sin(-baseLon) + cos(lat)*sin(lon)*cos(-baseLon) - sin(lat)*sin(-baseLon)*sin(-baseLat),
		zz = -cos(lat)*cos(lon)*sin(-baseLat) + sin(lat)*cos(-baseLat);
	lat = RAD2DEG(asin(zz)), lon = RAD2DEG(atan2(yy, xx));
}
#endif
