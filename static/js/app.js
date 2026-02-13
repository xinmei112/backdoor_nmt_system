async function fetchJSON(url){
  const res = await fetch(url, {cache:"no-store"});
  if(!res.ok) throw new Error("HTTP " + res.status);
  return await res.json();
}

function setText(id, text){
  const el = document.getElementById(id);
  if(el) el.textContent = text;
}

function setHTML(id, html){
  const el = document.getElementById(id);
  if(el) el.innerHTML = html;
}

function badgeForStatus(status){
  const s = (status || "").toLowerCase();
  if(s === "completed") return `<span class="badge ok"><span class="dot"></span>completed</span>`;
  if(s === "running") return `<span class="badge warn"><span class="dot"></span>running</span>`;
  if(s === "failed") return `<span class="badge bad"><span class="dot"></span>failed</span>`;
  return `<span class="badge"><span class="dot"></span>${status || "unknown"}</span>`;
}

async function pollJob(jobId, opts){
  const statusElId = opts?.statusElId || "jobStatus";
  const badgeElId = opts?.badgeElId || "jobBadge";
  const logElId = opts?.logElId || "jobLog";
  const canEvalElId = opts?.canEvalElId || "canEval";

  async function tick(){
    try{
      const data = await fetchJSON(`/api/job/${jobId}`);
      setText(statusElId, data.status || "");
      setHTML(badgeElId, badgeForStatus(data.status));

      if(canEvalElId){
        const can = (data.status === "completed" && data.best_model_path);
        const el = document.getElementById(canEvalElId);
        if(el) el.style.display = can ? "block" : "none";
      }

      // log tail
      if(logElId){
        const log = await fetchJSON(`/api/job/${jobId}/log?max_lines=200`);
        const el = document.getElementById(logElId);
        if(el){
          el.textContent = log.text || "";
          // keep scrolled to bottom if user hasn't scrolled up
          if(el.scrollTop + el.clientHeight + 40 >= el.scrollHeight){
            el.scrollTop = el.scrollHeight;
          }
        }
      }
    }catch(e){
      // ignore transient
    }
  }

  await tick();
  const interval = setInterval(tick, 2000);
  return () => clearInterval(interval);
}

window.__pollJob = pollJob;
