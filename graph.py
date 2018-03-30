import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
from collections import defaultdict

from graph_tool.all import *

def session_draw(cluster_id, sessions_id, log):
    for id in sessions_id:
        session = log[log.global_session_id==id]
        urls = session.requested_url
        urls = urls.append(log.referrer_url)
        urls.drop_duplicates(inplace=True)
        g = Graph()
        v = {}
        for u in urls:
            v[u] = g.add_vertex()
        session.apply(lambda x: g.add_edge(v[x.referrer_url], v[x.requested_url]), axis=1)
        graph_draw(g, output="Graphs/"+str(cluster_id)+"_session"+str(id)+".png")
    return;