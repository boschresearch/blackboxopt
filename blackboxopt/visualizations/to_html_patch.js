/* Copyright (c) 2020 - for information on the respective copyright owner
  see the NOTICE file and/or the repository https://github.com/boschresearch/blackboxopt

  SPDX-License-Identifier: Apache-2.0 */

// Patch to be injected in html output of Pythons `plotly.io.to_html()`.
//
// Features:
// - Toggle persistence hover boxes on click on data point
// - Copy value to clipboard on click on text in hover box

document
  .querySelector("div.plotly-graph-div.js-plotly-plot")
  .on("plotly_click", function (data) {
    let hoverLayer = document.querySelector("svg.main-svg > g.hoverlayer");
    let persistentHoverLayer = document.getElementById("persistent-hoverlayer");

    if (!persistentHoverLayer) {
      // Create container node for persistent version of the hover elements
      persistentHoverLayer = document.createElementNS(
        "http://www.w3.org/2000/svg",
        "g"
      );
      persistentHoverLayer.setAttribute("id", "persistent-hoverlayer");
      persistentHoverLayer.setAttribute(
        "style",
        "pointer-events: all; user-select: text; cursor:pointer;"
      );
      persistentHoverLayer.addEventListener("click", function (evt) {
        // Copy value to clipboard on click
        if ((evt.target.nodeName == "tspan") & evt.target.classList.contains("line")) {
          const text = evt.target.innerHTML;
          const delimIdx = text.indexOf(":");
          const value = text.substr(delimIdx + 1).trim();
          navigator.clipboard.writeText(value);
        }
      });
      hoverLayer.parentNode.append(persistentHoverLayer);
    }

    if (hoverLayer) {
      // Toggle persistent hover layer content
      if (persistentHoverLayer.innerHTML == hoverLayer.innerHTML) {
        persistentHoverLayer.innerHTML = "";
      } else {
        persistentHoverLayer.innerHTML = hoverLayer.innerHTML;
      }
    }
  });
