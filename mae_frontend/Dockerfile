# mae_frontend/ est une seule page statique (React + Babel charges depuis un
# CDN, aucune etape de build) : nginx sert directement index.html/mae_logo.png,
# pas besoin d'une image node.
#
#   docker build -f mae_frontend/Dockerfile -t mae-frontend mae_frontend
#   docker run -p 3000:80 mae-frontend
FROM nginx:alpine
COPY . /usr/share/nginx/html
EXPOSE 80
