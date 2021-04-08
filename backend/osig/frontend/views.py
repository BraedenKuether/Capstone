from django.shortcuts import render

# Create your views here.
def index(request):
  return render(request, 'frontend/index.html')
  
def view_run(request, id):
  request.GET = request.GET.copy()
  request.GET['id'] = id
  return render(request, 'frontend/index.html')