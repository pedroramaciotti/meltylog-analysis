import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
from collections import defaultdict

from graph_tool.all import *

def session_draw(filename, gsid, log):
    log = log[log.global_session_id==gsid]
    urls = log.requested_url
    urls = urls.append(log.referrer_url)
    urls.drop_duplicates(inplace=True)
    g = Graph()
    v = defaultdict(lambda: g.add_vertex())
    log.apply(lambda x: g.add_edge(v[x.referrer_url], v[x.requested_url]), axis=1)
    graph_draw(g, output=filename)
    return;