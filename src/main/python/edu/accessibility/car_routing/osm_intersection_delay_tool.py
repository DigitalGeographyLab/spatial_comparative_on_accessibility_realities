# -*- coding: utf-8 -*-
"""
MetropAccess-Digiroad 2.0: Creating a modified street network for more realistic car travel time routings. 

Requirements:
    geopandas 
    pandas
    networkx
    ogr
    geos=3.4.2

Important:
    The version of geos needs to be 3.4.2 because the prepared geometries are producing false results in newer GEOS versions --> see https://github.com/Toblerity/Shapely/issues/519.
    
    Install the older version of GEOS (3.4.2):
        conda install -c scitools geos=3.4.2

Last updated: 
    28.4.2018 

Author: 
    Henrikki Tenkanen
"""

import geopandas as gpd
import pandas as pd
import networkx as nx
import os
import osmnx as ox

from osgeo import ogr
from shapely.geometry import Point, LineString, MultiLineString, MultiPolygon, MultiPoint
from shapely import ops
from shapely.wkt import loads
from shapely import prepared
from fiona.crs import from_epsg
import matplotlib.pyplot as plt
from codes.DORA.digiroad_time_penalties import penalties as pns
from codes.DORA.src.connection.PostgresServiceProvider import PostGISServiceProvider
from codes.DORA.src.util import GPD_CRS, GeometryType

counter = 1


def process_speed_limits(link_gdf, road_type_col='highway', known_speed_limit_col='maxspeed',
                         speed_limit_col='KmH', length_col='new_length', freeflow_col='freeflow'):
    # Join the speed limits to links
    # data = link_gdf.merge(limit_gdf[['SEGM_ID', 'VAIK_SUUNT', 'ARVO']], on='SEGM_ID', how='outer')
    data = link_gdf.copy(deep=True)  # link_gdf.merge(limit_gdf, on='osmid', how='outer')

    # Remove 'ajopolku', i.e. paths that are made with tractor etc.
    # data = data.loc[data[road_type_col] != 7]

    # Remove road type '99' undefined
    # data = data.loc[data[road_type_col] != 99]

    # Add columns names for required fields 
    data[speed_limit_col] = None
    data[length_col] = None
    data[freeflow_col] = None

    print("Specify speed limits ..")
    # Specify speed limits for all road segments (add speed limit information based on the road type if segment does not contain the information)
    data = data.apply(associateSpeedLimitInformation, speed_limit_col=known_speed_limit_col,
                      road_type_col=road_type_col, target_col=speed_limit_col, axis=1)
    data.crs = link_gdf.crs  # after the apply, geodataframe is lossing his crs

    # Calculate the length of the geometries
    data[length_col] = data.length

    # Calculate the travel time based on speed limits 
    # ==> How many minutes does it to cross the road segment based on speed limit
    data[freeflow_col] = data[length_col] / (data[speed_limit_col] * 16.66666667)

    # Remove geometries without data
    # print(data.columns)
    data = data.dropna(subset=['geometry'])
    return data


def associateSpeedLimitInformation(row, speed_limit_col, road_type_col, target_col, walking_speed=4):
    """Function to associate speed limit information for all road segments either based on the information about the known speed limit, or based on the assumed speed limit based on road type. """
    # Use known information about speed limit if available
    if row["osmid"] == pd.np.nan:
        row["osmid"] = -1 * counter
        counter += 1
    if row[speed_limit_col] > 0:
        row[target_col] = row[speed_limit_col]
    else:
        # if row[road_type_col] == 1:
        #     row[target_col] = 90
        # elif row[road_type_col] == 2:
        #     row[target_col] = 80
        # elif row[road_type_col] == 3 or row[road_type_col] == 4:
        #     row[target_col] = 50
        # elif row[road_type_col] == 5 or row[road_type_col] == 6:
        #     row[target_col] = 40
        # elif row[road_type_col] == 8:
        #     row[target_col] = walking_speed
        if (row[road_type_col] == "trunk"):
            row[target_col] = 90
        elif row[road_type_col] == "primary":
            row[target_col] = 70
        elif row[road_type_col] == "secondary" \
                or (row[road_type_col] == "trunk_link"):
            row[target_col] = 60
        elif (row[road_type_col] == "tertiary") \
                or (row[road_type_col] == "tertiary_link") \
                or (row[road_type_col] == "primary_link") \
                or (row[road_type_col] == "secondary_link"):
            row[target_col] = 50

        elif (row[road_type_col] == "residential") or (row[road_type_col] == "road") \
                or (row[road_type_col] == "unclassified"):
            row[target_col] = 40
        elif row[road_type_col] == "living_street" or (row[road_type_col] == "yes"):
            row[target_col] = 20
        # elif row[road_type_col] == 8:
        else:
            row[target_col] = walking_speed
    return row


def edges_from_line(geom, attrs):
    """
    Generate edges for each line in geom
    Written as a helper for read_shp

    Parameters
    ----------

    geom:  ogr line geometry
        To be converted into an edge or edges

    attrs:  dict
        Attributes to be associated with all geoms

    Returns
    -------
     edges:  generator of edges
        each edge is a tuple of form
        (node1_coord, node2_coord, attribute_dict)
        suitable for expanding into a networkx Graph add_edge call
    """

    if geom.GetGeometryName().startswith('MULTI'):
        print("Multilinestring")

    edge_attrs = attrs.copy()
    last = geom.GetPointCount() - 1
    edge_attrs["Wkt"] = geom.ExportToWkt()
    yield (geom.GetPoint_2D(0), geom.GetPoint_2D(last), edge_attrs)


def generateGraphFromDigiroadShape(filepath, crs, name="unnamed", strict=True):
    """Generates nx.DiGraph() object from Digiroad 2.0. 
    
    Digiroad 2.0 cannot be read directly with nx.read_shp() function because of custom datatypes. """

    net = nx.DiGraph()
    net.graph["crs"] = crs
    net.graph["name"] = name
    shp = ogr.Open(filepath)
    if shp is None:
        raise RuntimeError("Unable to open {}".format(filepath))
    for lyr in shp:
        fields = [x.GetName() for x in lyr.schema]
        for f in lyr:
            g = f.geometry()
            if g is None:
                if strict:
                    raise nx.NetworkXError("Bad data: feature missing geometry")
                else:
                    continue
            flddata = [f.GetField(f.GetFieldIndex(x)) for x in fields]
            attributes = dict(zip(fields, flddata))
            attributes["ShpName"] = lyr.GetName()

            # Add x and y coordinates of the starting node
            attributes['x'] = g.GetPoint_2D(0)[0]
            attributes['y'] = g.GetPoint_2D(0)[1]

            # Digiroad 2.0 specific GeometryType is somekind of number always
            if isinstance(g.GetGeometryType(), int):
                # Add nodes
                net.add_node((g.GetPoint_2D(0)), **attributes)
                # Get edges
                for edge in edges_from_line(g, attributes):
                    e1, e2, attr = edge
                    net.add_edge(e1, e2)
                    net[e1][e2].update(attr)
            else:
                if strict:
                    raise nx.NetworkXError("GeometryType {} not supported".
                                           format(g.GetGeometryType()))
    return net


def generate_graph_from_Digiroad_GeoDataFrame(gdf, strict=True):
    """Generates nx.DiGraph() object from Digiroad 2.0 that is stored in a GeoDataFrame. 
    
    Digiroad 2.0 cannot be read directly with nx.read_shp() function because of custom datatypes. """

    print("TODO")


def get_nodes(graph):
    """Converts graph nodes into a GeoDataFrame."""
    # Get edges and nodes
    G = graph.copy()
    nodes = {node: data for node, data in G.nodes(data=True)}
    gdf_nodes = gpd.GeoDataFrame(nodes).T
    gdf_nodes.crs = G.graph["crs"]

    # Get coordinates from index
    gdf_nodes = gdf_nodes.reset_index()
    gdf_nodes = gdf_nodes.rename(columns={'level_0': 'x', 'level_1': 'y'})
    gdf_nodes['geometry'] = gdf_nodes.apply(lambda row: Point(row['x'].iloc[0], row['y'].iloc[0]), axis=1)
    return gdf_nodes


def get_edges(graph):
    """Converts graph edges into a GeoDataFrame."""
    G = graph.copy()

    # Get edges
    edges = []
    for u, v, data in G.edges(data=True):

        # for each edge, add key and all attributes in data dict to the
        # edge_details
        edge_details = {}
        for attr_key in data:
            edge_details[attr_key] = data[attr_key]

        if len(edge_details) > 0:
            # Create geometry
            try:
                edge_details['geometry'] = loads(data['Wkt'])
            except:
                pass
            # Remove unnecessary information
            del edge_details['Wkt']
            edges.append(edge_details)

    # create a GeoDataFrame from the list of edges and set the CRS
    gdf_edges = gpd.GeoDataFrame(edges)
    gdf_edges.crs = G.graph["crs"]
    return gdf_edges


def build_path(G, node, endpoints, path):
    """
    Recursively build a path of nodes until you hit an endpoint node.
    Parameters
    ----------
    G : networkx multidigraph
    node : int
        the current node to start from
    endpoints : set
        the set of all nodes in the graph that are endpoints
    path : list
        the list of nodes in order in the path so far
    Returns
    -------
    paths_to_simplify : list
    """
    # for each successor in the passed-in node
    for successor in G.successors(node):
        if successor not in path:
            # if this successor is already in the path, ignore it, otherwise add
            # it to the path
            path.append(successor)
            if successor not in endpoints:
                # if this successor is not an endpoint, recursively call
                # build_path until you find an endpoint
                path = build_path(G, successor, endpoints, path)
            else:
                # if this successor is an endpoint, we've completed the path,
                # so return it
                return path

    if (path[-1] not in endpoints) and (path[0] in G.successors(path[-1])):
        # if the end of the path is not actually an endpoint and the path's
        # first node is a successor of the path's final node, then this is
        # actually a self loop, so add path's first node to end of path to
        # close it
        path.append(path[0])

    return path


def is_endpoint(G, node):
    """
    Return True if the node is a "real" endpoint of an edge in the network, \
    otherwise False. OSM data includes lots of nodes that exist only as points \
    to help streets bend around curves. An end point is a node that either: \
    1) is its own neighbor, ie, it self-loops. \
    2) or, has no incoming edges or no outgoing edges, ie, all its incident \
        edges point inward or all its incident edges point outward. \
    3) or, it does not have exactly two neighbors and degree of 2 or 4. \
    4) or, if strict mode is false, if its edges have different OSM IDs. \
    Parameters
    ----------
    G : networkx multidigraph
    node : int
        the node to examine
    strict : bool
        if False, allow nodes to be end points even if they fail all other rules \
        but have edges with different OSM IDs
    Returns
    -------
    bool
    """
    neighbors = set(list(G.predecessors(node)) + list(G.successors(node)))
    n = len(neighbors)
    d = G.degree(node)

    if node in neighbors:
        # if the node appears in its list of neighbors, it self-loops. this is
        # always an endpoint.
        return True

    # if node has no incoming edges or no outgoing edges, it must be an endpoint
    elif G.out_degree(node) == 0 or G.in_degree(node) == 0:
        return True

    elif not (n == 2 and (d == 2 or d == 4)):
        # else, if it does NOT have 2 neighbors AND either 2 or 4 directed
        # edges, it is an endpoint. either it has 1 or 3+ neighbors, in which
        # case it is a dead-end or an intersection of multiple streets or it has
        # 2 neighbors but 3 degree (indicating a change from oneway to twoway)
        # or more than 4 degree (indicating a parallel edge) and thus is an
        # endpoint
        return True
    else:
        # if none of the preceding rules returned true, then it is not an endpoint
        return False


def get_paths_to_simplify(G, strict=True):
    """
    Create a list of all the paths to be simplified between endpoint nodes.
    The path is ordered from the first endpoint, through the interstitial nodes,
    to the second endpoint.
    Parameters
    ----------
    G : networkx multidigraph
    strict : bool
        if False, allow nodes to be end points even if they fail all other rules
        but have edges with different OSM IDs
    Returns
    -------
    paths_to_simplify : list
    """

    # first identify all the nodes that are endpoints
    endpoints = set([node for node in G.nodes() if is_endpoint(G, node)])

    paths_to_simplify = []

    # for each endpoint node, look at each of its successor nodes
    for node in endpoints:
        for successor in G.successors(node):
            if successor not in endpoints:
                # if the successor is not an endpoint, build a path from the
                # endpoint node to the next endpoint node
                try:
                    path = build_path(G, successor, endpoints, path=[node, successor])
                    paths_to_simplify.append(path)
                except RuntimeError:
                    print('Recursion error: exceeded max depth, moving on to next endpoint successor')
                    # recursion errors occur if some connected component is a
                    # self-contained ring in which all nodes are not end points
                    # handle it by just ignoring that component and letting its
                    # topology remain intact (this should be a rare occurrence)
                    # RuntimeError is what Python <3.5 will throw, Py3.5+ throws
                    # RecursionError but it is a subtype of RuntimeError so it
                    # still gets handled
    return paths_to_simplify


def simplify_graph(G, strict=True, speed_limit_col='KmH', length_col='length'):
    """
    Simplify a graph's topology by removing all nodes that are not intersections
    or dead-ends.
    Create an edge directly between the end points that encapsulate them,
    but retain the geometry of the original edges, saved as attribute in new
    edge.
    Parameters
    ----------
    G : networkx multidigraph
    strict : bool
        if False, allow nodes to be end points even if they fail all other rules
        but have edges with different OSM IDs
    Returns
    -------
    networkx multidigraph
    """

    G_temp = G.copy()
    all_nodes_to_remove = []
    all_edges_to_add = []

    # construct a list of all the paths that need to be simplified
    paths = get_paths_to_simplify(G_temp)

    for idx, path in enumerate(paths):
        edge_attributes = {}

        for u, v in zip(path[:-1], path[1:]):
            edge = G_temp.edges[u, v]
            for key in edge:
                # If value does not exits in the dict add it inside a list
                if not key in edge_attributes:
                    edge_attributes[key] = [edge[key]]
                else:
                    # If the key already exists, add it into the list
                    edge_attributes[key].append(edge[key])
        # Reduce the amount of values in lists by storing only unique values (if one, store value instead of a list)
        # Except speed limit and lenght columns
        for key, values in edge_attributes.items():
            if not key in [speed_limit_col, length_col]:
                unique = list(set(values))
                if len(unique) == 1:
                    edge_attributes[key] = unique[0]
                else:
                    edge_attributes[key] = unique
            else:
                edge_attributes[key] = values

        # construct the geometry and calculate the total length of the merged segments        
        edge_attributes['geometry'] = ops.linemerge(
            MultiLineString([loads(line_wkt) for line_wkt in edge_attributes['Wkt']]))

        # add the nodes and edges to their lists for processing at the end
        all_nodes_to_remove.extend(path[1:-1])

        all_edges_to_add.append({'origin': path[0],
                                 'destination': path[-1],
                                 'attr_dict': edge_attributes})
    # for each edge to add in the list we assembled, create a new edge between
    # the origin and destination
    for edge in all_edges_to_add:
        G_temp.add_edge(edge['origin'], edge['destination'], **edge['attr_dict'])

    # finally remove all the interstitial nodes between the new edges
    G_temp.remove_nodes_from(set(all_nodes_to_remove))

    return G_temp


def convertListsToStr(df):
    """Convert list values in df to strings so that they can be saved to Shapefile"""
    # Convert lists in edges to strings
    for col in df.columns:
        for value in df[col]:
            if isinstance(value, list):
                df[col] = df[col].astype(str)
                break
    return df


def calculate_node_connections(df, graph):
    """Calculate the number of connections from node which can be used to determine if the node is an intersection"""
    # Create column for the number of connections
    df['connections'] = None

    # Find out in and out-degree
    for x, data in graph.nodes(data=True):
        connections = sum([graph.out_degree(x), graph.in_degree(x)])
        try:
            # Update connection to nodes
            df.loc[df['osmid'] == data['osmid'], 'connections'] = connections
        except:
            pass

    df['connections'] = df['connections'].fillna(1)
    return df


def get_list_rows(df, column):
    """Method that identifies and returns rows that have list as a value"""
    lists_or_strings = df[pd.to_numeric(df[column], errors='coerce').isnull()]
    lists = lists_or_strings[lists_or_strings[column].apply(lambda x: type(x) == list)]
    return lists


def sum_list_rows(df, column):
    """
    If a row value includes a list of values, this function sums them. If the value is something else than a list, it will leave the value as it is. 
    Possible NaN values in a list will be ignored. 
    """
    df[column] = df[column].apply(
        lambda value: sum(list(filter(None.__ne__, value))) if isinstance(value, list) else value)
    return df


def geom_touch(gdf, geometry):
    """
    Select features from GeoDataFrame that touches the geometry.

    Parameters
    ----------

    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing the features that will be compared against the geometry
    geometry : shapely.geometry
        Shapely geometry that is used to select the data (if the geometry touches the feature)

    Returns
    -------
    geopandas.GeoDataFrame containing the selected rows

    """
    print(gdf.head())
    mask = gdf.touches(geometry)
    result = gdf.loc[mask]
    return result


def parse_speed_limits(gdf, speed_column='KmH', length_column='new_length'):
    """Harmonizes the speed limits if multiple values exist after simplifying the road segments. Use the speed limit of the longest road segment"""

    # Get speed and lenght values
    gdf['speedlimit'] = None

    for idx, row in gdf.iterrows():
        if isinstance(row[speed_column], list):
            # Get unique
            u = list(set(row[speed_column]))
            if len(u) == 1:
                gdf.loc[idx, 'speedlimit'] = u[0]
            else:
                # Determine the speed limit of the longest road segment
                # Get the index of the longest road segment
                index = row[length_column].index(max(row[length_column]))
                gdf.loc[idx, 'speedlimit'] = row[speed_column][index]
        else:
            gdf.loc[idx, 'speedlimit'] = row[speed_column]
    return gdf


def prepare_geometry(gdf, geom_col):
    """Create prepared geometries"""
    # Detect geometry type
    gtype = gdf.head(1)[geom_col].values[0].geom_type

    if gtype in ['LineString', 'MultiLineString']:
        prepared_geom = prepared.prep(MultiLineString(list(gdf[geom_col].values)))
    elif gtype in ['Polygon', 'MultiPolygon']:
        prepared_geom = prepared.prep(ops.cascaded_union(MultiPolygon(list(gdf[geom_col].values))))
    else:
        raise Exception(
            "Geometries should be either LineStrings or Polygons (or geometry collection of them). Got: %s" % gtype)
    return prepared_geom


def fast_intersect(left_gdf, right_gdf, left_geom_col='geometry', right_geom_col='geometry', prepare_right=True):
    """Make fast spatial intersect to large number of points by using prepared geometries. Assumes that all geometries in right_gdf are of same type (LineString | Polygon) """

    # Create prepared geometry
    if not prepare_right:
        prepared_geom = prepare_geometry(left_gdf, geom_col=left_geom_col)
        hits = gpd.GeoDataFrame(list(filter(prepared_geom.intersects, list(right_gdf[right_geom_col].values))),
                                columns=['geometry'], crs=right_gdf.crs)
        result = gpd.sjoin(right_gdf, hits)
    else:
        prepared_geom = prepare_geometry(right_gdf, geom_col=right_geom_col)
        # Find out the intersecting geometries
        hits = gpd.GeoDataFrame(list(filter(prepared_geom.intersects, list(left_gdf[left_geom_col].values))),
                                columns=['geometry'], crs=left_gdf.crs)
        result = gpd.sjoin(left_gdf, hits)
    return result


def fast_contains(left_gdf, right_gdf, left_geom_col='geometry', right_geom_col='geometry'):
    """Make fast contains query to large number of points by using prepared geometries. Assumes that all geometries in right_gdf are of same type (LineString | Polygon) """

    # Create prepared geometry
    prepared_geom = prepare_geometry(right_gdf, geom_col=right_geom_col)

    # Find out the intersecting geometries
    hits = gpd.GeoDataFrame(list(filter(prepared_geom.contains, list(left_gdf[left_geom_col].values))),
                            columns=['geometry'], crs=left_gdf.crs)
    result = gpd.sjoin(left_gdf, hits)
    return result


def func_ramp_intersections(row, penalties, rtype, avg='avg', midd='m', rush='r', median='median', freeflow='freeflow',
                            speed_limit='KmH', jtype3='jtype3', jtype5='jtype5', a_target='Kokopva_aa',
                            m_target='Keskpva_aa', r_target='Ruuhka_aa'):
    """ Process slip roads: 
        
        - slow speed slip roads that intersect with another road
        - high speed slip roads that intersect with another road
        - high speed slip roads
        - slow speed slip roads
    """
    # CHECKED
    if row[jtype3] == 1 and row[jtype5] != 2:
        if row[speed_limit] < 70:
            if rtype == 'rt456':
                row[a_target] = row[freeflow] + (penalties[avg][rtype] / 3) * 2
                row[m_target] = row[freeflow] + (penalties[midd][rtype] / 3) * 4
                row[r_target] = row[freeflow] + (penalties[rush][rtype] / 2)
            else:
                row[a_target] = row[freeflow] + (penalties[avg][rtype] / 3)
                row[m_target] = row[freeflow] + (penalties[midd][rtype] / 3)
                row[r_target] = row[freeflow] + (penalties[rush][rtype] / 2)
        else:
            row[a_target] = row[freeflow] + (penalties[avg][rtype] / 3)
            row[m_target] = row[freeflow] + (penalties[midd][rtype] / 4)
            row[r_target] = row[freeflow] + (penalties[rush][rtype] / 2)

    # High speed slip roads
    elif row[speed_limit] >= 70:
        # Multiply with the median delay
        row[a_target] = row[freeflow] * penalties[avg][median]
        row[m_target] = row[freeflow] * penalties[midd][median]
        row[r_target] = row[freeflow] * penalties[rush][median]
    else:
        row[a_target] = row[freeflow] + ((penalties[avg][rtype] / 3) * 2)
        row[m_target] = row[freeflow] + ((penalties[midd][rtype] / 3) * 4)
        row[r_target] = row[freeflow] + ((penalties[rush][rtype] / 2) * 2)
    return row


def func_normal_high_speed_intersection(row, penalties, avg='avg', midd='m', rush='r', freeflow='freeflow',
                                        speed_limit='KmH', jtype5='jtype5', a_target='Kokopva_aa',
                                        m_target='Keskpva_aa', r_target='Ruuhka_aa'):
    """ 'Normal' intersections that can be driven with high speed"""
    # CHECKED
    if row[jtype5] == 0 and row[speed_limit] >= 70:
        row[a_target] = row[freeflow] * 1.1
        row[m_target] = row[freeflow]
        row[r_target] = row[freeflow] * 1.2
    return row


def func_signal_intersections(row, penalties, rtype, avg='avg', midd='m', rush='r', freeflow='freeflow',
                              speed_limit='KmH', jtype1='jtype1', jtype5='jtype5', a_target='Kokopva_aa',
                              m_target='Keskpva_aa', r_target='Ruuhka_aa'):
    """ 
    Intersections with traffic lights or two intersecting roads under same road class
    """
    # CHECKED
    if row[jtype1] == 1 or row[jtype5] == 2:
        row[a_target] = row[freeflow] + penalties[avg][rtype]
        row[m_target] = row[freeflow] + penalties[midd][rtype]
        row[r_target] = row[freeflow] + penalties[rush][rtype]
    return row


def func_roundabout_intersections(row, penalties, rtype, element_type, roundabout_code, avg='avg', midd='m',
                                  rush='r', freeflow='freeflow', speed_limit='KmH', jtype1='jtype1', jtype5='jtype5',
                                  a_target='Kokopva_aa', m_target='Keskpva_aa', r_target='Ruuhka_aa'):
    """ Roundabout intersections """
    # CHECKED
    # if (row["junction"] in roundabout_code) and (row[jtype5] != 0):
    row[a_target] = row[freeflow] + ((penalties[avg][rtype] / 3) * 2)
    row[m_target] = row[freeflow] + (penalties[midd][rtype] / 2)
    row[r_target] = row[freeflow] + ((penalties[rush][rtype] / 4) * 3)
    return row


def func_normal_intersection(row, penalties, rtype, avg='avg', midd='m', rush='r', freeflow='freeflow',
                             jtype2='jtype2', jtype4='jtype4', jtype5='jtype5',
                             a_target='Kokopva_aa', m_target='Keskpva_aa', r_target='Ruuhka_aa'):
    """ Normal intersection"""
    # CHECKED
    if row[jtype2] == 1 or row[jtype4] == 1 and row[jtype5] != 2:
        row[a_target] = row[freeflow] + (penalties[avg][rtype] / 2)
        row[m_target] = row[freeflow] + (penalties[midd][rtype] / 2)
        row[r_target] = row[freeflow] + (penalties[rush][rtype] / 2)
    return row


def func_normal_slow_intersecting_intersection(row, penalties, rtype, element_type, slip_road_code,
                                               avg='avg', midd='m', rush='r', freeflow='freeflow', speed_limit='KmH',
                                               jtype3='jtype3', jtype4='jtype4', jtype5='jtype5', a_target='Kokopva_aa',
                                               m_target='Keskpva_aa', r_target='Ruuhka_aa'):
    """Normal (slow) road that intersects with another one """
    # CHECKED
    if row[jtype3] == 1 and row[speed_limit] < 70 and row[element_type] not in slip_road_code and row[jtype5] != 2:
        row[a_target] = row[freeflow] + (penalties[avg][rtype] / 3)
        row[m_target] = row[freeflow] + (penalties[midd][rtype] / 4)
        row[r_target] = row[freeflow] + (penalties[rush][rtype] / 2)
    return row


def func_other_intersections(row, penalties, rtype, avg='avg', midd='m', rush='r', freeflow='freeflow',
                             a_target='Kokopva_aa', m_target='Keskpva_aa', r_target='Ruuhka_aa'):
    """All other kind of intersections"""
    # CHECKED
    row[a_target] = row[freeflow] + (penalties[avg][rtype] / 4)
    row[m_target] = row[freeflow] + (penalties[midd][rtype] / 4)
    row[r_target] = row[freeflow] + (penalties[rush][rtype] / 4)
    return row


def func_process_penalties(row, penalties, rtype, element_type, slip_road_code, roundabout_code,
                           jtype1='jtype1', jtype2='jtype2', jtype3='jtype3', jtype4='jtype4', jtype5='jtype5',
                           speed_limit='KmH'):
    """Process the fast road classes, i.e. road classes 1 and 2 in Digiroad 2.0 """
    # Start with slip roads
    if row[element_type] in slip_road_code:
        row = func_ramp_intersections(row=row, penalties=penalties, rtype=rtype)

    # "Normal" intersections that can be driven with high speed
    elif row[jtype5] == 0 and row[speed_limit] >= 70:
        row = func_normal_high_speed_intersection(row=row, penalties=penalties, rtype=rtype)

    # Intersections with traffic lights or two intersecting roads under same road class
    elif row[jtype1] == 1 or row[jtype5] == 2:
        row = func_signal_intersections(row=row, penalties=penalties, rtype=rtype)

    # Roundabout intersections 
    elif row["junction"] in roundabout_code and row[jtype5] != 0:
        row = func_roundabout_intersections(row=row, penalties=penalties, rtype=rtype, element_type=element_type,
                                            roundabout_code=roundabout_code)

    # Normal (slow) road that intersects with another one 
    elif (row[jtype3] == 1) and (row[speed_limit] < 70) and (row[element_type] not in slip_road_code) and (
        row[jtype5] != 2):
        row = func_normal_slow_intersecting_intersection(
            row=row, penalties=penalties, rtype=rtype,
            element_type=element_type,
            slip_road_code=slip_road_code
        )

    # Normal intersection
    elif row[jtype2] == 1 or row[jtype4] == 1 and row[jtype5] != 2:
        row = func_normal_intersection(row=row, penalties=penalties, rtype=rtype)

    # All other cases
    else:
        row = func_other_intersections(row=row, penalties=penalties, rtype=rtype)
    return row


def func_drivethrough(row, freeflow='freeflow', a_target='Kokopva_aa', m_target='Keskpva_aa', r_target='Ruuhka_aa'):
    """Use the free flow travel time."""
    row[a_target] = row[freeflow]
    row[m_target] = row[freeflow]
    row[r_target] = row[freeflow]
    return row


def assign_intersection_penalties(row, penalties, road_class,
                                  element_type, slip_road_code, roundabout_code,
                                  avg='avg', midd='m', rush='r', median='median', r_target='Ruuhka_aa',
                                  m_target='Keskpva_aa', a_target='Kokopva_aa', jtype1='jtype1',
                                  jtype2='jtype2', jtype3='jtype3', jtype4='jtype4', jtype5='jtype5',
                                  freeflow='freeflow', speed_limit='KmH'):
    """Helper function to be used with pandas apply function for fast iterations"""

    # if row[road_class] in [1, 2]:
    #     row = func_process_penalties(row=row, penalties=penalties, rtype='rt12')
    #
    # elif row[road_class] == 3:
    #     row = func_process_penalties(row=row, penalties=penalties, rtype='rt3')
    #
    # elif row[road_class] in [4, 5, 6]:
    #     row = func_process_penalties(row=row, penalties=penalties, rtype='rt456')
    # else:
    #     row = func_drivethrough(row=row)
    if row[road_class] in ["trunk", "trunk_link", "primary", "secondary"]:
        row = func_process_penalties(
            row=row, penalties=penalties, rtype='rt12',
            element_type=element_type,
            slip_road_code=slip_road_code,
            roundabout_code=roundabout_code
        )

    elif row[road_class] in ["tertiary", "tertiary_link", "primary_link", "secondary_link"]:
        row = func_process_penalties(
            row=row, penalties=penalties, rtype='rt3',
            element_type=element_type,
            slip_road_code=slip_road_code,
            roundabout_code=roundabout_code
        )

    elif row[road_class] in ["residential", "road", "unclassified", "living_street", "yes"]:
        row = func_process_penalties(
            row=row, penalties=penalties, rtype='rt456',
            element_type=element_type,
            slip_road_code=slip_road_code,
            roundabout_code=roundabout_code
        )
    else:
        row = func_drivethrough(row=row)
    return row


def calculate_penalties(gdf, penalties, avg='avg', midd='m', rush='r', median='median', r_target='Ruuhka_aa',
                        m_target='Keskpva_aa', a_target='Kokopva_aa', road_class='highway',
                        element_type='highway', slip_road_code=6, roundabout_code=5, jtype1='jtype1',
                        jtype2='jtype2', jtype3='jtype3', jtype4='jtype4', jtype5='jtype5', freeflow='freeflow',
                        speed_limit='KmH'):
    """
    Calculates the drive through times for all road segments. 
    
    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame having all the Digiroad 2.0 street edges. Columns 'jtype1', 'jtype2', 'jtype3', 'jtype4', 'jtype5', 'freeflow', and 'TOIMINN_LK' needs to existing and´determined before this function works.
    penalties : dict
        A dictionary that contains information about the intersection penalties for different road types. 
    penalty_dict : dict
        
    """
    data = gdf.apply(assign_intersection_penalties, penalties=penalties,
                     avg=avg, midd=midd, rush=rush, median=median,
                     r_target=r_target, m_target=m_target, a_target=a_target,
                     road_class=road_class, freeflow=freeflow, speed_limit=speed_limit,
                     element_type=element_type,
                     slip_road_code=slip_road_code, roundabout_code=roundabout_code,
                     jtype1=jtype1, jtype2=jtype2, jtype3=jtype3, jtype4=jtype4, jtype5=jtype5,
                     axis=1)
    data.crs = gdf.crs

    return data


def main():
    # =================
    # File paths
    # =================
    # def main():
    data_folder = r"C:\Users\jeisonle\repository\HelsinkiRegionTravelTimeMatrix2018\data\accessibility"
    link_fp = os.path.join(data_folder, "shapefiles", "tallinn_all_edges.shp")
    limits_fp = os.path.join(data_folder, "shapefiles", "tallinn_all_edges.shp")
    signals_fp = os.path.join(data_folder, "shapefiles", "tallinn_all_nodes.shp")

    # =============================
    # Parameters & attribute names
    # =============================

    # Coordinate reference system of the source data
    epsg_code = 3067

    # Walking speed (km/h)
    walking_speed = 4

    # Existing attributes
    known_speed_limit_column = 'maxspeed'
    road_type_column = 'highway'

    # New attributes that will be created
    allocated_speed_limit_column = 'KmH'
    length_column = 'new_length'  # length (in meters)
    freeflow_column = 'freeflow'

    # =================
    # Read files
    # =================

    print("Read files ..")
    links = gpd.read_file(link_fp)
    limits = gpd.read_file(limits_fp)
    signals = gpd.read_file(signals_fp)

    links = links.to_crs({'init': 'epsg:3067'})
    limits = limits.to_crs({'init': 'epsg:3067'})
    signals = signals.to_crs({'init': 'epsg:3067'})

    links["old_osmid"] = links["osmid"]
    limits["old_osmid"] = limits["osmid"]
    signals["old_osmid"] = signals["osmid"]

    links["osmid"] = range(1, len(links) + 1, 1)
    limits["osmid"] = range(1, len(limits) + 1, 1)
    signals["osmid"] = range(1, len(signals) + 1, 1)

    # remove not drivable nor walking paths
    links = links.loc[
        ~(links["highway"] == "cycleway") &
        ~(links["highway"] == "service")
        ]
    links = links[["osmid", "old_osmid", "highway", "junction", "length", "oneway", "maxspeed", "geometry"]]

    # limits = limits.loc[
    #     ~(limits["highway"] == "cycleway") &
    #     ~(limits["highway"] == "service")
    #     ]
    # limits = limits[["osmid", "maxspeed", "geometry"]]
    links[known_speed_limit_column] = pd.to_numeric(links[known_speed_limit_column])

    signals = signals.loc[signals["highway"] == "traffic_signals"]
    signals = signals[["osmid", "old_osmid", "highway", "geometry"]]

    # ===============================
    # Assign speed limit information
    # ===============================

    data = process_speed_limits(link_gdf=links,
                                # limit_gdf=limits,
                                known_speed_limit_col=known_speed_limit_column,
                                speed_limit_col=allocated_speed_limit_column,
                                road_type_col=road_type_column)

    # Separate pedestrian paths from the road network
    pedestrian_edges = data.loc[
        (data[allocated_speed_limit_column] == walking_speed) |
        (data["highway"] == "pedestrian") |
        (data["highway"] == "footway") |
        (data["highway"] == "steps") |  # stairs
        (data["highway"] == "bridleway")  # horses and walkers
        ]
    data = data.loc[~data.index.isin(pedestrian_edges.index)]

    # Save data without walking parts as a temporary file
    temp_fp = link_fp.replace('.shp', '_temp.shp')
    data.to_file(temp_fp)

    print("Generate graph..")
    # Generate graph from the roads 
    G1 = generateGraphFromDigiroadShape(temp_fp, data.crs)
    west, south, east, north = 24.522766, 59.311105, 25.012344, 59.57849  # Tallinn, Estonia, same as the geojson 'links_fp'

    networkType = 'all'
    G = ox.graph_from_bbox(north, south, east, west, network_type=networkType, simplify=False)
    G_projected = ox.project_graph(G)

    # print("Simplify graph ..")
    # # Generate graph with only intersection and ending nodes and the edges between them
    sgraph1 = simplify_graph(G1)
    sgraph = G.copy()
    sgraph = ox.simplify_graph(sgraph)
    sgraph_temp = sgraph.copy()

    # Get nodes and edges
    sn = get_nodes(sgraph1)
    se = get_edges(sgraph1)
    # sn, se = ox.save_load.graph_to_gdfs(sgraph1, nodes=True, edges=True,
    #                                     node_geometry=True, fill_edge_geometry=True)

    # Calculate the number of connections
    sn = calculate_node_connections(df=sn, graph=sgraph1)

    # Select intersections
    intersections = sn.loc[sn['connections'] > 1]

    print("Detect traffic lights ..")
    # Detect intersections that are within 20 meters from traffic signals
    signals['buffer'] = signals.buffer(20)
    signals = signals.set_geometry('buffer')

    # Find out which intersections are affected by traffic signals (prepared geometries produce false results for some reason)
    signal_intersections = fast_intersect(intersections, signals, right_geom_col='buffer')

    # Find out which edges touch the points
    signal_geom = MultiPoint(list(signal_intersections['geometry'].values))
    signal_edges = data.loc[data.touches(signal_geom)]

    print("Detect slip roads ..")
    # Extract the locations of slip road edges (interchange | ramp)
    slip_road_edges = data.loc[
        (data['highway'] == "motorway_link") |
        (data['highway'] == "trunk_link") |
        (data['highway'] == "primary_link") |
        (data['highway'] == "secondary_link") |
        (data['highway'] == "tertiary_link")
        ]

    print("Detect other edges ..")
    # Select all other types of edges 
    assigned_edges = list(set(list(pedestrian_edges.index) + list(signal_edges.index) + list(slip_road_edges.index)))
    other_edges = data.loc[~data.index.isin(assigned_edges)]

    # Assign flags for different junction types
    signal_edges = signal_edges.assign(jtype1=1)
    other_edges = other_edges.assign(jtype2=1)
    slip_road_edges = slip_road_edges.assign(jtype3=1)
    pedestrian_edges = pedestrian_edges.assign(jtype4=1)

    # Combine and assign junction types for all edges
    join = data.merge(signal_edges[['jtype1', 'osmid']], on="osmid", how="left")
    join = join.merge(other_edges[['jtype2', 'osmid']], on='osmid', how='left')
    join = join.merge(slip_road_edges[['jtype3', 'osmid']], on='osmid', how='left')
    join = join.merge(pedestrian_edges[['jtype4', 'osmid']], on='osmid', how='outer')

    # Fill NaN in junction types with 0
    jtype_cols = ['jtype1', 'jtype2', 'jtype3', 'jtype4']
    join[jtype_cols] = join[jtype_cols].fillna(0)
    join['jtype5'] = join[jtype_cols].sum(axis=1)

    # ===============================================================
    # Calculate drive through times using the intersection penalties
    # ===============================================================
    print("Calculate drive through times using intersection penalties ..")
    result = calculate_penalties(
        gdf=join, penalties=pns, element_type="highway",
        slip_road_code=["motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link"],
        roundabout_code=["roundabout"]
    )

    # ==========================
    # Save to disk
    # ==========================
    print("Saving edges to disk .. ")
    outfp = link_fp.replace('.shp', '_intersection_delayed.shp')
    result = result.loc[result['geometry'].notnull()]
    result.to_file(outfp)

    # # Save to database
    # postGISServiceProvider = PostGISServiceProvider()
    # isExecuted = postGISServiceProvider.insert(
    #     result,
    #     tableName="tallinn_road_network",
    #     isTableExist="replace",
    #     geometryType=GeometryType.POLYGON
    # )

    # ===========================
    # Simplify network if wanted
    # ===========================

    G = generateGraphFromDigiroadShape(outfp, result.crs)

    print("Simplify the result graph for faster routing..")
    # Generate graph with only intersection and ending nodes and the edges between them
    sgraph = simplify_graph(G)

    # Get nodes and edges
    sn = get_nodes(sgraph)
    se = get_edges(sgraph)
    # sn, se = ox.save_load.graph_to_gdfs(sgraph, nodes=True, edges=True,
    #                                     node_geometry=True, fill_edge_geometry=True)

    # Harmonize the speed limits. Use the speed limit of the longest road segment with longest distance
    # se = parse_speed_limits(se, length_column='new_length')

    # Sum the values of the travel times
    se = sum_list_rows(se, column=freeflow_column)
    se = sum_list_rows(se, column='Keskpva_aa')
    se = sum_list_rows(se, column='Kokopva_aa')
    se = sum_list_rows(se, column='Ruuhka_aa')

    # Calculate the total distance of the merged road segments
    se[length_column] = se.length

    # Remove unnecessary columns
    se = se.drop(['jtype1', 'jtype2', 'jtype3', 'jtype4', 'jtype5', 'ShpName'], axis=1)

    # Put the new columns at the end of the file
    cols = se.columns
    cols_to_end = ['KmH', 'new_length', 'freeflow', 'Kokopva_aa', 'Keskpva_aa', 'Ruuhka_aa']
    new_order = [col for col in cols if col not in cols_to_end]
    new_order = new_order + cols_to_end
    se = se[new_order]

    # Convert lists to strings
    se = convertListsToStr(se)

    # Save to disk
    outfp = outfp.replace('.shp', '_simplified.shp')
    # se = se.to_crs(GPD_CRS.PSEUDO_MERCATOR)
    se.to_file(outfp)

    # # Save to database
    # isExecuted = postGISServiceProvider.insert(
    #     se,
    #     tableName="tallinn_road_network_simplified",
    #     isTableExist="replace",
    #     geometryType=GeometryType.POLYGON
    # )
