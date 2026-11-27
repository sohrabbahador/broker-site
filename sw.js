const C='ba-v8';
self.addEventListener('install',e=>{e.waitUntil(caches.open(C).then(c=>c.addAll(['./','index.html','manifest.webmanifest','icon-192.png','icon-512.png'])));self.skipWaiting()});
self.addEventListener('activate',e=>{e.waitUntil(caches.keys().then(k=>Promise.all(k.filter(x=>x!==C).map(x=>caches.delete(x)))));self.clients.claim()});
self.addEventListener('fetch',e=>{
 const req=e.request;
 if(req.method!=='GET')return;
 let url;try{url=new URL(req.url)}catch(err){return}
 // فقط فایل‌های همین سایت کش می‌شوند؛ درخواست‌های GitHub API و سرویس‌های بیرونی دست‌نخورده می‌مانند
 if(url.origin!==self.location.origin)return;
 e.respondWith(fetch(req).then(r=>{
  if(r&&r.ok&&r.status===200&&r.type==='basic'){const cp=r.clone();caches.open(C).then(c=>c.put(req,cp)).catch(()=>{})}
  return r;
 }).catch(()=>caches.match(req).then(m=>m||caches.match('./'))));
});
